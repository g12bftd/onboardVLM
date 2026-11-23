# Design

## 0. The teacher is the exact Gemma 4 E2B / E4B vision encoder

* **Pinned checkpoints.** `teachers/hub.py` lists the four supported checkpoints at pinned commits.
* **Vision-only download.** `fetch_vision_weights` reads the safetensors header and fetches only
  the `model.vision_tower.*` and `model.embed_vision.*` byte ranges (about 320 MB). It hashes the
  raw checkpoint bytes and compares them with `manifests/gemma4_vision.json`.
* **Construction.** `build_vision_tower` builds `Gemma4VisionModel` with `_from_config(dtype=bf16)`,
  exactly like `from_pretrained`, and loads with `strict=True`.
* **Precision guard.** RoPE `inv_freq` stays fp32. Every module holding it re-asserts fp32 after
  `.to(dtype)` (`restore_rope_fp32`); rounding it to bf16 changes the outputs.
* **Load-time check.** `Gemma4Teacher` re-hashes every loaded tensor against the manifest (`verify`).
* **Full check.** `scripts/verify_gemma4_teacher.py --full_check` compares against
  `Gemma4ForConditionalGeneration.from_pretrained`: weights, collator inputs vs the official
  processor, `encode_vision` vs `get_image_features`, and `project_to_lm` vs `embed_vision`.
  All are bitwise equal at 70, 280 and 1,120 tokens.

The manifest shows that E2B, E4B, base and `-it` share one byte-identical vision tower; only
`embed_vision` differs. "Distilling the E2B or E4B encoder" is therefore one distillation target.
The teacher config only decides which `embed_vision` (1536-d or 2560-d LM space) is used for
LM-space losses and which VLM is used in Stage 2 and evaluation.

## 1. Unified vision output contract

Every encoder, teacher or student, returns packed and unpadded tokens:

* `tokens [N, 768]`, `intermediate_tokens {layer: [N, D_layer]}`, `image_token_lengths`.
* Images are concatenated in batch order.
* Within an image, tokens are row-major over its own `(h/48, w/48)` grid.

This is exactly `Gemma4VisionModel`'s order: its pooler uses kernel index
`x//3 + (W_patches//3) * (y//3)` and strips padding with a boolean mask.

## 2. Geometry and degradations happen once

`geometry.sample_transform` returns a serialisable `TransformParams`: crop, flip, target size,
budget, and capture degradations. `apply_transform` runs the crop, then the flip, then the Gemma
resize (uint8, bicubic, antialias; bit-identical to `Gemma4ImageProcessor`), then the
degradations. The single uint8 result is branched into two views:

* **Teacher view**: Gemma patches padded to `max_soft_tokens * 9`.
* **Student view**: a dense batch with `valid_token_mask`.

The teacher therefore labels exactly the pixels the student sees ("consistent teaching"; tested).
Stage 2 and evaluation use the official processor for text, fed the transformed images with
`do_resize=False`.

## 3. Students

| Student | What it is | Input |
|---|---|---|
| `gemma4_vit` (`patch_size: 16`) | transformers' own `Gemma4VisionModel` with configurable depth, width, heads, MLP size, position-table size and clamps; optional weight-selection init from the exact teacher | the teacher's patch sequence, as-is; dense images also accepted |
| `gemma4_vit` (`patch_size: 48`) | same blocks with one 48-px patch per soft token and no pooling (9× fewer tokens); random init | dense images |
| `timm_spatial` | any timm CNN / hybrid with exact stride 48: `stem_stride3` surgery or a stride-16 tap + 3×3/s3 conv, verified at construction | dense images |
| `compact_vit`, `coarse_patch_vit`, `multiplexed_vit` | from-scratch ViTs (2D-RoPE, MiniViT weight sharing) | dense images |

**Weight-selection init** (`students/gemma4_vit.py`, Xu et al. ICLR 2024):

* Keep `num_hidden_layers` teacher layers (`uniform`, `first`, `last` or an explicit list).
* Inside each layer, keep whole 64-d heads (`uniform`, `first`, or `magnitude` = ||W_o||).
* Keep MLP channels (`uniform`, `first`, or `magnitude` = ||W_down[:, j]|| · ||W_up[j]||).
* Keep a residual-dimension subset that is consistent across *all* tensors (`uniform`, `first`,
  or `magnitude` = mean RMSNorm gain), and the first `position_embedding_size` rows of the 2D
  position table.
* When `hidden_size < 768`, the output projection is initialised to scatter the selected
  dimensions back to their teacher positions, scaled by sqrt(768/hidden) to undo the pooler's
  sqrt(D) factor.
* A full-size copy reproduces the teacher bitwise (tested).

**Alignment proof for CNNs.** With circular padding, a 48-px input shift moves the token map by
exactly one token. A stride-32 network plus interpolation fails this test.

**Batch invariance.** ViT-type students mask padding, so batched and per-image outputs agree to
fp32 rounding. CNNs see their neighbours' zero padding, so the grafted tower runs them per image.

## 4. Losses (YAML-composable, `config/criterion/*.yaml`)

* `token_feature`: MSE, Smooth-L1 or cosine, after RMS normalisation by default.
* `relational_token`: per-image token-similarity matching; never a global Gram matrix.
* `intermediate_layer`: on the layers in `layer_map`. Teacher layers are pooled with Gemma's own
  position rule (tested identical to `Gemma4VisionPooler`).

Each term has `space: vision | lm`:

* `vision` (default) compares the 768-d tower outputs.
* `lm` compares the tokens after the teacher checkpoint's frozen `embed_vision` (scale-free
  RMSNorm + projection, computed in fp32): the exact space the language model reads.

## 5. Pipeline

**Stage 1 (encoder-only).** The frozen teacher tower runs online, or its outputs come from the
safetensors cache. The adapter adds projection heads only where a student's width differs from
768 (intermediate layers of narrow or CNN students). The adapter is never exported.

**Stage 2 (optional).** `GraftedVLM` swaps `vlm.model.vision_tower` for a `StudentVisionTower`,
whose call signature matches `Gemma4VisionModel`. Images and video frames both go through it.

* Default: the student is trainable and **everything else in the VLM is frozen and stays
  byte-identical** (tested).
* Target logits come from the same VLM with its original tower (`teacher_vision()` context).
* KL is computed on answer spans only, with `logits_to_keep` so the 262K-vocabulary tensor is
  only materialised for the answer tail.
* Unfreezing `embed_vision` (full / LoRA) is an ablation.
* A different logit teacher (e.g. E4B logits for an E2B deployment) is approximate: the two
  tokenizers differ in 1 vocabulary entry and 14 merges.

## 6. Reproducibility

* Pinned checkpoint commits (`config/teacher/*.yaml` and `SUPPORTED_TEACHERS`).
* Per-tensor hashes, checked at every load.
* Every random augmentation is stored in `TransformParams` (cache replay is exact).
* Per-worker RNGs; Hydra configs saved with every run.
