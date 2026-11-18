# onboard-vlm

Code for **"Toward Fully Onboard Vision-Language Models for Smart Glasses"**: distilling the
vision encoders of Gemma 4 (E2B / E4B / 31B) into lightweight students that can be **grafted
back into the Gemma 4 VLM** and evaluated with the unchanged Hugging Face `.generate()` path on
egocentric and VQA benchmarks.

* PyTorch + `transformers>=5.4` (`Gemma4*`), `timm`, `safetensors`, Hydra/OmegaConf, PyTorch-Lightning.
* Configuration, data and training layout follow TimeFM `cerebro_v2`: Hydra module configs +
  `+experiment=...` overrides, LMDB datasets (`<split>.lmdb` + `<split>.meta.json`) built by
  `make_datasets/lmdb/make_*.py`, Lightning tasks.

```
onboardvlm/
├── onboard_vlm/
│   ├── interfaces.py          # StudentVisionOutput / TeacherVisionOutput / DistillBatch / TransformParams / ABCs
│   ├── registry.py            # @register_teacher / _student / _loss / _dataset + build_*()
│   ├── geometry.py            # Gemma 4 resize/patchify (bit-exact w/ HF), dense padding, unpatchify
│   ├── data/                  # LMDB I/O, datasets, DualViewCollator, VLMEvalCollator, prompts, samplers, DataModule
│   ├── teachers/              # Gemma4Teacher (hooks, vision-only loading), safetensors feature cache
│   ├── students/              # CompactViT, CoarsePatchViT, MultiplexedViT, TimmSpatialStudent, io
│   ├── distill/               # DistillationAdapter, losses, Lightning tasks (Stage 1 / Stage 2)
│   ├── vlm/grafted.py         # GraftedVLM + StudentVisionTower (hot-swap vlm.model.vision_tower)
│   ├── eval/                  # official metrics, LLM judges, benchmark specs, runner
│   └── utils/                 # profiler, tiny offline Gemma 4 (tests / dry runs)
├── config/                    # Hydra: teacher/ student/ criterion/ task/ data_module/ trainer/ experiment/ benchmark/
├── make_datasets/             # download/*.sh (raw) -> lmdb/make_*.py (LMDB)
├── run_distill.py  run_eval.py  run_cache_teacher.py  run_profile.py
├── docs/                      # design.md, datasets.md, metrics.md
└── tests/                     # 89 tests, CPU-only, no downloads
```

## Install

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

Gemma 4 weights are gated on the Hub: run `hf auth login` and accept the license first.

## Quick start

Offline sanity check (random tiny Gemma 4, synthetic images, CPU):

```bash
python -u run_distill.py +experiment=debug/dryrun_stage1
```

```bash
python run_eval.py benchmark=synthetic_debug tiny=true device=cpu dtype=float32 use_judge=false student_dir=outputs/students/dryrun_stage1
```

Data (see [docs/datasets.md](docs/datasets.md) for details and per-dataset notes):

```bash
RAW_ROOT=/scratch/raw bash make_datasets/download/download_all.sh
```

```bash
RAW_ROOT=/scratch/raw LMDB_ROOT=/scratch/lmdb bash make_datasets/build_all_lmdb.sh
```

Stage 1 (feature distillation from the frozen Gemma 4 vision tower) and Stage 2 (graft into the
VLM, logit KD, only `embed_vision`/LoRA trainable):

```bash
ONBOARD_VLM_DATA=/scratch/lmdb python -u run_distill.py +experiment=stage1/compact_vit_s_e2b
```

```bash
ONBOARD_VLM_DATA=/scratch/lmdb python -u run_distill.py +experiment=stage2/compact_vit_s_e2b
```

Optional teacher-feature cache for fast Stage-1 sweeps:

```bash
python run_cache_teacher.py teacher=gemma4_e2b dataset.path=/scratch/lmdb/vizwiz cache_root=/scratch/cache/vizwiz_e2b 'layers=[3,9,15]'
```

```bash
python -u run_distill.py +experiment=stage1/compact_vit_s_e2b data_module=stage1_cached cache_root=/scratch/cache/vizwiz_e2b
```

Evaluation (teacher baseline, then a grafted student; `OPENAI_API_KEY` is needed for the
LLM-judged benchmarks):

```bash
python run_eval.py -m benchmark=vizwiz_vqa,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema_subset,cvbench
```

```bash
python run_eval.py -m benchmark=vizwiz_vqa,textvqa,cvbench student_dir=outputs/students/stage2_compact_vit_s_e2b run_name=cvit_s
```

Profiling (params, MACs, peak activation memory at 140/280/560 tokens):

```bash
python run_profile.py device=cuda dtype=bfloat16 latency_iters=50
```

## Core contracts (see [docs/design.md](docs/design.md))

```python
@dataclass
class StudentVisionOutput:            # TeacherVisionOutput has the same layout
    tokens: Tensor                    # [N, D_target], N = sum(image_token_lengths), unpadded, Gemma order
    image_token_lengths: list[int]
    intermediate_tokens: dict[int, Tensor]   # {layer: [N, D_layer]} on the same 48-px grid

class VisionStudent(nn.Module):
    out_dim: int; layer_dims: dict[int, int]; batch_invariant: bool
    def forward(self, images: Tensor,             # [B, 3, H, W] in [0,1], H, W multiples of 48
                valid_token_mask: Tensor,         # [B, H/48, W/48]
                return_layers=()) -> StudentVisionOutput: ...

class VisionTeacher:
    def encode_vision(self, inputs: TeacherInputs, layers=()) -> TeacherVisionOutput: ...
    def forward_vlm(self, **vlm_inputs) -> Tensor: ...   # [B, L, V] logits

@dataclass
class DistillBatch:                   # emitted by DualViewCollator
    teacher_inputs: TeacherInputs     # pixel_values [B, P, 768], pixel_position_ids [B, P, 2]
    student_images: Tensor            # [B, 3, H, W]
    valid_token_mask: Tensor          # [B, H/48, W/48]
    image_token_lengths: list[int]
    transforms: list[TransformParams]
    text_inputs: dict | None          # Stage 2: input_ids, attention_mask, mm_token_type_ids, kd_mask ...
    cached_teacher: TeacherVisionOutput | None
```

## Where the brief was adjusted (verified against `transformers` 5.23 source)

| Brief | What Gemma 4 actually does / what the code does |
|---|---|
| Processor "pads images to multiples of 48" and outputs flattened unpadded tokens | `Gemma4ImageProcessor` **resizes** (floor) to the largest `H, W` multiples of 48 within `max_soft_tokens * 9` patches (it also *upsamples* small images), outputs `pixel_values [B, max_soft_tokens*9, 768]` padded along the *patch sequence* with position `-1`, plus `image_position_ids` and `num_soft_tokens_per_image`. The **vision tower output** is the flattened, unpadded `[N, D]`. Our geometry reproduces the processor bit-exactly (tested). Spatial padding only exists in the dense student batch (bottom/right, whole 48-px cells). |
| `pixel_position_ids` | Called `image_position_ids` by the processor / VLM forward and `pixel_position_ids` by `Gemma4VisionModel`; `TeacherInputs.vlm_kwargs()` / `.vision_kwargs()` handle both. |
| Collator emits *normalised* student images | Gemma uses no mean/std (it maps `[0,1]` to `[-1,1]` inside the patch embedder). Students therefore receive `[0,1]` images and normalise internally (`InputNorm`: `gemma` or `imagenet` for timm weights); that keeps the training path and the grafted path (which reconstructs `[0,1]` images from processor patches) identical. |
| Intermediate teacher tokens `[N, D]` | Gemma's encoder layers run on 16-px patches (9x the output tokens). Intermediate states are pooled 3x3 with Gemma's own position-based rule (tested identical to `Gemma4VisionPooler`), so final and intermediate tokens share one order. |
| EgoBlind "multi-reference VQA accuracy" | Official metric is a **GPT-4o-mini judge** given all 4 references -> accuracy (% "yes") + mean 0-5 score per question type (`EgoBlind/eval.py`). Implemented with the verbatim prompt; deterministic proxies are reported separately. |
| EgoTextVQA "ANLS + accuracy" | Official metric is a **GPT-4o-mini judge** -> accuracy + mean 0-5 score per question type (`gpt_eval.py`). ANLS/EM are kept as clearly-labelled auxiliary metrics. The high/low-resolution study applies to **Outdoor only**. |
| "Gemma-4-12B Unified, encoder-free" | Not among the variants documented in transformers (E2B, E4B, 26B-A4B, 31B). The teacher rejects any config with `vision_config is None` (authoritative) and fails fast on `gemma-4-12b` ids. |
| 31B vision tower = 27 layers x 1152 | Could not be verified offline; nothing is hard-coded: widths/depths are read from `config.vision_config` and the `DistillationAdapter` sizes its heads from them. |

## Tests

```bash
pytest
```

All tests are CPU-only and offline (tiny random Gemma 4 + an offline `Gemma4Processor`):
geometry parity with `Gemma4ImageProcessor`, teacher/student shape & unpadding parity for all
four families across 6 aspect ratios x 3 budgets, exact 48x alignment (one-token shift
equivariance + a negative control for stride-32 + interpolation), multiplexed parameter
sharing, per-image relational isolation, grafted `generate()` for images and videos, official
metric implementations, the LMDB builders on synthetic raw data, the teacher cache, and
Stage-1/Stage-2 Lightning steps.
