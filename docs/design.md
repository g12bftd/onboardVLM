# Design

## 0. The teacher is the original Gemma 4 E2B / E4B model

* **Original weights, original code.** `teachers/gemma4.py::load_gemma4_vlm` calls
  `Gemma4ForConditionalGeneration.from_pretrained(model_id, revision=...)`, as any Gemma 4 user
  would. `Gemma4Teacher` keeps `vlm.model.vision_tower` (167M params) and `vlm.model.embed_vision`.
  With `load: vision`, the default for feature and embedding distillation, it releases the language
  model. Nothing is re-implemented, re-hashed or patched.
* **Supported checkpoints.** `SUPPORTED_TEACHERS` lists `google/gemma-4-E2B-it`, `-E4B-it`, `-E2B`
  and `-E4B`, each with a default pinned commit for reproducibility. Any other id raises, including
  the encoder-free 12B "unified" model. A different `revision` can be passed explicitly.
* **One encoder.** The vision tower is the same in all four checkpoints (658 tensors,
  167,364,608 params; verified byte-for-byte in commit 6b24885). Only `embed_vision` differs:
  768→1536 for E2B, 768→2560 for E4B. "Distilling the E2B or E4B encoder" is therefore one vision
  target. The teacher choice decides which `embed_vision` the embedding task uses, and which VLM the
  logit task and evaluation use.

## 1. Unified vision output contract

Every encoder, teacher or student, returns packed and unpadded tokens:

* `tokens [N, 768]`, `intermediate_tokens {layer: [N, D_layer]}`, `image_token_lengths`.
* Images are concatenated in batch order.
* Within an image, tokens are row-major over its own `(h/48, w/48)` grid.

This is exactly `Gemma4VisionModel`'s order: its pooler uses kernel index
`x//3 + (W_patches//3) * (y//3)` and strips padding with a boolean mask.

## 2. Geometry and degradations happen once

`geometry.sample_transform` returns a serialisable `TransformParams`: crop, flip, target size,
budget, and capture degradations. `apply_transform` runs, in order:

1. the crop;
2. the flip;
3. the Gemma resize (uint8, bicubic, antialias; bit-identical to `Gemma4ImageProcessor`);
4. the degradations.

The single uint8 result is branched into two views:

* **Teacher view**: Gemma patches padded to `max_soft_tokens * 9`.
* **Student view**: a dense batch with `valid_token_mask`.

The teacher therefore labels exactly the pixels the student sees ("consistent teaching"; tested).
The logit task and evaluation use the official processor for text, fed the transformed images with
`do_resize=False`.

## 3. Students

| Student | What it is | Input |
|---|---|---|
| `gemma4_vit` | transformers' own `Gemma4VisionModel`, built from a smaller `Gemma4VisionConfig` (depth, width, heads, head dim, MLP size, position table, clamps), randomly initialised by transformers. A linear head maps `hidden_size` to 768. | the teacher's patch sequence, as-is; dense images also accepted |
| `timm_spatial` | any timm CNN / hybrid / efficient ViT on the exact 48-px token grid (see below) | dense images |
| `compact_vit`, `coarse_patch_vit`, `multiplexed_vit` | from-scratch ViTs (2D-RoPE, MiniViT weight sharing) | dense images |
| `gemma4_truncated` (baseline) | the original tower cut to its first *k* layers + its own pooler; zero-training baseline or initialisation | the teacher's patch sequence |

Any `timm_spatial` student can freeze its pretrained backbone (`freeze_backbone: true`, BatchNorm
statistics included) so only the grid-alignment / projection head trains: the "off-the-shelf
encoder + projector" baseline. Students with learned position tables must cover the declared
input domain (`check_student_covers`); see [protocol.md](protocol.md#3-input-domain).

### Putting a CNN on Gemma's token grid

Gemma emits one token per 48×48-px cell, in row-major order over an `(H/48, W/48)` grid. A per-token
linear / MLP head only changes the *channel* dimension. It cannot create the right *number* of
tokens, nor put each token on the right 48-px cell:

* A stride-32 CNN gives `(H/32) × (W/32)` cells, 2.25× too many, and they straddle Gemma's cells.
  Resampling them (pooling, bilinear) blurs neighbouring cells together and breaks the one-token
  shift test below.
* Fixed-resolution models (LeViT, EfficientViT-M, EfficientFormerV2, MobileViTv2) fail *before* the
  head: they reject any input that is not 224×224, or their position biases are tied to a 7×7 grid.

`TimmSpatialStudent` therefore has four `alignment` modes. All of them are verified at construction
on probe sizes:

| mode | how | backbone | used for |
|---|---|---|---|
| `input_rescale` (default) | resize the image by r/48 (2/3 for stride-32 nets, area filter), so stride r lands exactly on 48-px cells | **unmodified** pretrained weights | MobileNetV3/V4, EfficientNet(V2), FastViT, MobileCLIP(2), RepViT, StarNet, TinyViT, ConvNeXt-V2, EdgeNeXt, MambaOut, SHViT, ... |
| `stem_stride3` | change the first stride-2 conv to stride 3 (48 = 3 × 16) | one conv modified | ablation |
| `tap16_conv3` | take the stride-16 stage, add a 3×3/s3 conv | truncated | MobileNetV5 (Gemma 3n) |
| `resample` | resize each image to the model's fixed input, then a learned **position-query cross-attention head**: one query per output token, from Fourier features of its cell centre, attending over the backbone's feature map | unmodified | LeViT, EfficientViT-M, EfficientFormerV2, MobileViTv2 |

`input_rescale` keeps the "MLP at the end" idea, but applies it to a feature map that already lies
on the token grid. With an area filter, 2/3 downscaling maps every 48-px cell onto a 32-px cell
with no boundary special cases, so a 48-px input shift moves the token map by exactly one token
(tested with circular padding). A stride-32 net plus interpolation fails the same test (negative
control). `resample` is the honest version of "add a layer at the end" for fixed-resolution
models. It works at every grid size and is batch invariant. It cannot recover detail the backbone
never saw (224 px input), which is why those models land at ~1 GMAC (`run_profile.py`).

**Batch invariance.** Students that mask padding (ViTs, resample heads) give the same output batched
or one image at a time, up to fp32 rounding. Grid-aligned CNNs see their neighbours' zero padding,
so the grafted tower runs them one image at a time.

## 4. Losses (YAML-composable, `config/criterion/*.yaml`)

* `token_feature`: MSE, Smooth-L1 or cosine, after RMS normalisation by default.
* `relational_token`: per-image token-similarity matching; never a global Gram matrix.
* `intermediate_layer`: on the layers in `layer_map`. Teacher layers are pooled with Gemma's own
  position rule (tested identical to `Gemma4VisionPooler`).
* `grafted_logit_kd`: temperature-scaled KL on the answer span (+ optional CE).

`CompositeLoss` sums weighted terms and logs each one under its `label`.

## 5. Distillation types are tasks

Following TimeFM `cerebro_v2`, each distillation type is a Lightning task class. It is selected by
`config/task/<type>.yaml` and built as `hydra.utils.instantiate(cfg.task, cfg)`, i.e.
`Task(hparams, **task_cfg)`. `BaseDistillationTask` (`onboard_vlm/tasks/base_task.py`) holds what
they share:

* student construction;
* lazy teacher loading;
* AdamW with warmup-cosine;
* loss logging;
* checkpoint-safe batch transfer;
* export.

| task (`config/task/`) | what is compared | what is loaded | criterion configs |
|---|---|---|---|
| `embedding_distillation` (**default, main method**) | both token sets after the checkpoint's **frozen `embed_vision`**: the exact LM input space (1536-d / 2560-d) | vision tower + `embed_vision`; never the LM | `embedding_tokens`, `embedding_tokens_relational` (+ `task.vision_criterion` for vision-space terms) |
| `feature_distillation` (ablation) | student vs teacher **vision tokens** (768-d, RMS-normalised) and optional intermediate layers | vision tower only (or nothing, with an offline teacher cache) | `feature_tokens`, `feature_tokens_relational`, `feature_full` |
| `logit_distillation` (stage 2) | **next-token distributions** of the VLM with the student grafted in, vs the same VLM with its own tower; KL on the teacher's own responses (`run_generate_targets.py`) | the full VLM; only the student trains, so the VLM stays byte-identical (tested) | `logit_kd` |

The logit task starts from the embedding export of the same seed. Distilling on human reference
answers (`data_module=vqa_train`) and unfreezing `embed_vision` (full / LoRA) are ablations.

**Uniform evaluation across types.** Every task logs the same scale-invariant fidelity metrics on
val / test (`onboard_vlm/distill/fidelity.py`):

* `fid_lm_cos` and `fid_lm_rel_err` in the LM input space;
* `fid_vis_cos`, `fid_vis_cos_p05` and `fid_cka` in the vision space.

They are reported per (group, budget), per group, per budget and overall. Checkpoints of all three
types are selected on the same criterion, `val/fid_lm_cos`. The logit task also reports
`kd_top1_agree`.

## 6. Real-time logging

`onboard_vlm/utils/logging.py` sets up one logger tree (`onboard_vlm.*`). It writes to stdout and to
`<hydra run dir>/run_rank{r}.log`. Rank 0 logs INFO; other ranks log warnings only.

* `stage("...")` brackets every pipeline step: `===> name` when it starts, then
  `<=== name done in Xs | peak RSS / GPU memory` (or `FAILED`). Covered steps:
  * seeding;
  * building the task;
  * building the student;
  * downloading / loading the original Gemma 4 weights;
  * grafting;
  * setting up every dataset (with sizes and mixture weights);
  * building the trainer;
  * training;
  * validation;
  * export.
* `PipelineLoggingCallback` reports training progress:
  * at fit start: devices, strategy, precision and step budget;
  * every N steps: losses, lr, images/s, elapsed time, ETA and memory;
  * at every validation: start and end metrics;
  * at checkpoint saves and at the end of each epoch;
  * on exceptions.
* `run_cache_teacher.py`, `run_eval.py` and `run_profile.py` use the same stages.

## 7. Batching

Training uses one budget and one processor shape per batch (`BucketedBatchSampler`), with about
`tokens_per_batch` soft tokens and exact per-budget shares of batches. Evaluation uses one dataloader
per (group, budget), batched by exact processor shape (`ExactShapeBatchSampler`). No batch ever
contains padding, so CNN students behave at training exactly as at batch-1 deployment. All samplers
shard across DDP ranks themselves (`use_distributed_sampler: false`). Full rationale:
[protocol.md](protocol.md#4-training).

## 8. Reproducibility

* Pinned default checkpoint commits (`config/teacher/*.yaml`, `SUPPORTED_TEACHERS`).
* Every random augmentation is stored in `TransformParams`, so cache replay is exact.
* The corpus mixture sampler is deterministic given seed + epoch, and identical on all ranks.
* Corpus splits are a seeded content hash, so they are reproducible and stable across corpus sizes.
* The finalized corpus is identified by a digest (`docs/corpus.md`).
* The Hydra config is saved with every run and logged at start-up; `run_id = <experiment>_s<seed>`
  separates seeds; `train_summary.json` records the training compute of every run.
