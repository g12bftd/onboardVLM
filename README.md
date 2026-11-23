# onboard-vlm

Code for **"Toward Fully Onboard Vision-Language Models for Smart Glasses"**. It distils
**exactly the vision encoder of Gemma 4 E2B / E4B**, byte-for-byte the official pretrained weights,
into lightweight students. A student replaces the vision tower inside the *unchanged* Gemma 4 VLM,
so Hugging Face `.generate()` and every evaluator run identically for teacher and students.

Configuration, data and training follow TimeFM `cerebro_v2`:

* Hydra module configs plus `+experiment=...` overrides;
* LMDB datasets built by `make_datasets/lmdb/make_*.py`;
* PyTorch-Lightning tasks.

## The teacher, verified against the official checkpoints

| Fact | Evidence |
|---|---|
| Supported teachers, pinned commits: `google/gemma-4-E2B-it`, `-E4B-it`, `-E2B`, `-E4B` (Apache-2.0) | `onboard_vlm/teachers/hub.py::SUPPORTED_TEACHERS`; any other id or revision raises |
| The 658 vision-tower tensors (167,364,608 params, bf16, 319 MB) are byte-identical in all four checkpoints. There is one Gemma 4 small vision encoder. | `onboard_vlm/teachers/manifests/gemma4_vision.json` (SHA-256 of every tensor's raw checkpoint bytes) |
| Only `embed_vision.embedding_projection` (768→1536 for E2B, 768→2560 for E4B) differs, including base vs `-it` | same manifest |
| Our teacher matches HF `Gemma4ForConditionalGeneration.from_pretrained` bitwise: weights, collator inputs vs the official processor, vision tokens vs `get_image_features`, and LM embeddings vs `embed_vision`, at 70, 280 and 1,120 tokens | `scripts/verify_gemma4_teacher.py --full_check`; report in `docs/verification/gemma-4-E2B-it_full_check.json` |
| Only the vision tensors are fetched (HTTP range reads of the safetensors file, about 320 MB instead of 10–16 GB), and they are re-hashed against the manifest on every load | `hub.fetch_vision_weights`, `hub.verify_module_weights` |

Two defects were found and fixed while verifying:

* **Teacher targets (earlier commit):** building the tower in fp32 and calling `.to(bfloat16)` rounds the RoPE `inv_freq` buffer to bf16. Targets were off by up to 512 in absolute value (token magnitudes reach about 3.7e4; min token cosine 0.9994). The tower is now built as `from_pretrained` builds it, and every module that holds RoPE buffers re-asserts fp32 after any dtype cast.
* **transformers:** `Gemma4VisionModel` with `pooling_kernel_size=1` returns zero tokens (inverted pooler mask). Gemma 4 itself always pools 3×3, so this only affects our 48-px students, which use a local fix.

## What exactly is distilled

Only the vision encoder. Every student emits `[N, 768]` tokens in Gemma's order and plugs into the
checkpoint's own `embed_vision` and language model, which are never modified by default.

1. **Stage 1, encoder-only (main stage).** The teacher is just the 167M-param vision tower; the
   language model is never loaded. Losses can be computed in two spaces:
   * `space: vision`: the 768-d tower outputs, with RMS-normalised MSE + cosine, per-image token
     relations and layer-mapped intermediates;
   * `space: lm`: the same tokens after the checkpoint's frozen `embed_vision` (scale-free
     RMSNorm + projection). This is exactly what the LM consumes, at the cost of one matmul.

   `criterion=stage1_tokens_lm` (default) uses both. The choice of E2B vs E4B teacher only
   selects this projection, because the tower is identical.
2. **Stage 2, optional: LM-in-the-loop.** The student is grafted into the frozen VLM, which serves
   as a differentiable loss: KL on next-token logits over answer spans from benchmark train
   splits. Gradients update **only the student encoder**, so the deployed Gemma 4 stays
   byte-identical. Unfreezing `embed_vision` (full or LoRA) remains available as an ablation.

Use Stage 1 for every comparison (it is cheap and task-agnostic, and only 320 MB of teacher sits
on the GPU). Add Stage 2 to close the remaining gap on the VQA benchmarks. Feature matching cannot
tell which errors the LM is sensitive to; Stage 2 can, but it is task-specific and much costlier
(a full VLM forward and backward per step, plus a teacher VLM forward).

## Layout

```
onboard_vlm/
  interfaces.py  registry.py  geometry.py         # contracts, registries, bit-exact Gemma 4 geometry
  teachers/  hub.py gemma4.py feature_cache.py manifests/   # exact E2B/E4B teacher + offline cache
  students/  gemma4_vit.py timm_spatial.py vit.py multiplexed_vit.py io.py
  distill/   adapter.py losses.py tasks.py        # vision / LM-space losses, Lightning stages
  data/      collator.py degrade.py datasets.py hashing.py lmdb_io.py prompts.py samplers.py data_module.py
  vlm/       grafted.py                           # hot-swap vlm.model.vision_tower
  eval/      benchmarks.py runner.py metrics/     # official metrics, LLM judges, submissions
  utils/     profiler.py tiny.py
config/      teacher/ student/ criterion/ data_module/ task/ trainer/ experiment/{stage1,stage2,sweeps,debug} benchmark/
make_datasets/  download/*.sh  lmdb/make_*.py (benchmarks)  corpus/*.py (distillation corpora)
scripts/verify_gemma4_teacher.py
run_distill.py  run_eval.py  run_cache_teacher.py  run_profile.py
docs/  design.md  corpus.md  datasets.md  metrics.md  verification/
```

## Install and verify

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

```bash
python scripts/verify_gemma4_teacher.py --model_id google/gemma-4-E2B-it --full_check
```

The `--full_check` run downloads the full checkpoint (10 GB) and needs about 12 GB of RAM. Without
it, only the 320 MB vision tensors are fetched and verified.

## Running experiments (all YAML)

```bash
python -u run_distill.py +experiment=stage1/gemma4_vit_l8_d384_e2b
```

```bash
python -u run_distill.py +experiment=stage1/gemma4_vit_l8_d384_e2b teacher=gemma4_e4b student.num_hidden_layers=6 student.hidden_size=512
```

```bash
python -u run_distill.py -m +experiment=sweeps/gemma4_vit_depth_width
```

```bash
python -u run_distill.py -m +experiment=sweeps/cnn_zoo
```

```bash
python -u run_distill.py +experiment=stage2/grafted_kd_e2b stage1_experiment=stage1_gemma4_vit_l8_d384_e2b
```

```bash
python run_eval.py -m benchmark=vizwiz_vqa,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema_subset,cvbench student_dir=outputs/students/stage1_gemma4_vit_l8_d384_e2b run_name=g4vit_s
```

```bash
python -u run_distill.py +experiment=debug/dryrun_stage1
```

The last command is an offline sanity check (tiny random Gemma 4, synthetic data, CPU).

Hydra merges dicts, so replace one with `~layer_map '+layer_map={1: 2}'`. Writing `layer_map={...}`
only adds keys.

### Teacher (`config/teacher/`)

`gemma4_e2b` (default), `gemma4_e4b`, `gemma4_e2b_base` and `gemma4_e4b_base` each set the pinned
`revision`, `compute_dtype` and `verify: true`:

* `compute_dtype: bfloat16` is bitwise identical to the deployed VLM;
* `compute_dtype: float32` is a lossless upcast of the same weights.

### Students (`config/student/`)

Every family has one fully commented base config, and presets inherit from it. Any knob can be
overridden on the CLI. Profile below: 1080×1440 frame at 280 tokens (`run_profile.py`).

| Family | Presets | Knobs | GMACs | Params |
|---|---|---|---|---|
| **Gemma 4 teacher** (reference) | | 16 L × 768, 16-px patches | 539 | 167.4M |
| **`gemma4_vit`**: a smaller Gemma 4 encoder (transformers' `Gemma4VisionModel` code) | `gemma4_vit_l8_d768`, `_l12_d512`, `_l8_d512`, `_l8_d384` (default), `_l6_d256`, `_teacher_copy`, `_l8_d384_scratch` | `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `head_dim`, `intermediate_size`, `position_embedding_size`, `use_clipped_linears`, `init: teacher\|random`, `layer_selection`, `width_selection`, `head_selection`, `mlp_selection` | 253 / 192 / 129 / 81 / 33 | 76.5 / 51.4 / 34.6 / 19.7 / 6.8M |
| **`gemma4_vit` with `patch_size: 48`**: Gemma 4 blocks, one patch per token | `gemma4_vit_p48_l8_d384`, `_p48_l12_d512` | same knobs | 6.2 / 15.3 | 22.0 / 54.6M |
| **`timm_spatial`**: CNN / hybrid with exact stride 48 | MobileNetV4 conv S/M/L and hybrid M/L, MobileNetV5-300M (Gemma 3n encoder), MobileNetV3, EfficientNet B0/B2, EfficientNetV2 B0/B3/S, FastViT T12/SA12, MobileCLIP/MobileCLIP2 MCi0–3, RepViT, EfficientViT B1/B2/L1, GhostNetV3, StarNet, TinyViT 5M/11M, MobileOne, HGNetV2, ConvNeXt-V2 atto/nano, EdgeNeXt | `model_name` (any timm id + pretrained tag), `alignment: stem_stride3\|tap16_conv3`, `head`, `model_kwargs` | 3–33 (e.g. MNv4-M 4.7, MCi2 33, StarNet-S2 3.0) | 2–63M |
| `compact_vit` / `coarse_patch_vit` / `multiplexed_vit` | `compact_vit_s`, `coarse_patch_vit_s`, `multiplexed_vit_s(_lowrank)` | depth, width, heads, patch 16/48, MiniViT sharing | 104 / 7 / 106 | 22 / 24 / 8M |

`gemma4_vit` with `init: teacher` copies a subset of the exact teacher weights: chosen layers,
whole heads, and selected residual and MLP dimensions ("weight selection", Xu et al. ICLR 2024).
At full size the copy is bitwise identical to the teacher (tested). Pruned initialisations start
far from the teacher (vision-space cosine 0.18 for the depth-pruned L8-D768, about 0 when width is
sliced), so they are warm starts. The `init: teacher,random` sweep measures whether they help.

Seven timm families cannot be used, or need the `tap16_conv3` alignment:

* EfficientFormerV2, MobileViTv2 and MambaOut fail stride-48 alignment and are excluded (verified).
* ConvNeXt-V2, EdgeNeXt, FasterNet and InceptionNeXt have stride-4 patchify stems, so they use `tap16_conv3`.

### Data

* Benchmarks (download + LMDB): [docs/datasets.md](docs/datasets.md).
* Official metrics: [docs/metrics.md](docs/metrics.md).
* Distillation corpus (what to use, decontamination, builders): [docs/corpus.md](docs/corpus.md).

The benchmark train splits alone (about 45K images) are not enough to preserve the encoder's
general capabilities. `data_module=stage1_general` mixes general, high-resolution, text-rich,
egocentric and accessibility images, with capture degradations applied identically to teacher and
student.

## Other corrections to the original brief

| Brief | Reality (verified) |
|---|---|
| Processor pads images to multiples of 48 | It *resizes* (and upsamples small images) to multiples of 48 and pads only the patch sequence. Our geometry is bit-exact to the official processor. |
| Collator emits normalised images | Gemma uses no mean/std; students normalise internally. |
| EgoBlind / EgoTextVQA string metrics | Both use official GPT-4o-mini judges (prompts reproduced verbatim); see docs/metrics.md. |
| 12B "Unified" is encoder-free | Correct: transformers ships it as `gemma4_unified` (patches embedded directly into the LM). It has no vision tower and is rejected, like every non-E2B/E4B id. |
| E2B and E4B share a tokenizer | Almost: 1 of 262,144 vocabulary entries and 14 BPE merges differ. Cross-model logit KD is therefore flagged as approximate. |

## Tests

```bash
pytest
```

```bash
ONBOARD_VLM_HUB_TESTS=1 pytest tests/test_hub.py
```

The first command runs 200 tests, CPU-only and offline. They cover:

* geometry parity with `Gemma4ImageProcessor`;
* teacher/student shape parity for every family;
* exact stride-48 alignment;
* bitwise teacher inheritance, multiplexed parameter sharing and per-image relational isolation;
* LM-space losses, and Stage 2 leaving the VLM byte-identical;
* degradation replay and identical teacher/student pixels;
* corpus decontamination;
* every YAML preset and experiment, plus grafted `generate()` for images and video.

The second command adds the real-checkpoint tests (about 320 MB per checkpoint).
