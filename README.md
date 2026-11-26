# onboard-vlm

Code for **"Toward Fully Onboard Vision-Language Models for Smart Glasses"**. It distils the
**vision encoder of the original Gemma 4 E2B / E4B models** into lightweight students. The teacher
is the official checkpoint, loaded with `transformers`' own `from_pretrained`, unmodified. A student
replaces the vision tower inside the *unchanged* Gemma 4 VLM, so Hugging Face `.generate()` and
every evaluator run identically for teacher and students.

Configuration, data and training follow TimeFM `cerebro_v2`:

* Hydra module configs plus `+experiment=...` overrides;
* one Lightning **task per distillation type**;
* LMDB datasets;
* stage-by-stage real-time logging.

**The experimental protocol** (claims and evidence, student contract, input domain, training,
baselines, evaluation tiers, statistics, decontamination, ablations, run order) is in
[docs/protocol.md](docs/protocol.md).

## The teacher

`config/teacher/*.yaml` selects one of the four original checkpoints:

| config | model | default pinned revision |
|---|---|---|
| `gemma4_e2b` (default) | `google/gemma-4-E2B-it` | `4adf157` |
| `gemma4_e4b` | `google/gemma-4-E4B-it` | `4ecc2ba` |
| `gemma4_e2b_base` | `google/gemma-4-E2B` | `433813a` |
| `gemma4_e4b_base` | `google/gemma-4-E4B` | `11b8250` |

* The teacher is `Gemma4ForConditionalGeneration.from_pretrained(model_id, revision)` →
  `model.vision_tower` + `model.embed_vision`. The language model is released unless a task needs it.
* All four checkpoints share one vision tower (167.4M params, byte-identical). They differ only in
  `embed_vision` (768→1536 for E2B, 768→2560 for E4B) and in the language model.
* Any other model id raises (e.g. the encoder-free 12B "unified" model).

## Distillation types = tasks

| `task=` | compares | loads | role | example |
|---|---|---|---|---|
| `embedding_distillation` (default) | tokens after the checkpoint's frozen `embed_vision`: exactly what the LM reads | vision tower + `embed_vision` | **main method** | `+experiment=embedding/gemma4_vit_l8_d384_e2b` |
| `feature_distillation` | 768-d vision tokens (scale-invariant) + optional intermediate layers | vision tower only (none with an offline cache) | ablation | `+experiment=feature/gemma4_vit_l8_d384_e2b` |
| `logit_distillation` | next-token distributions of the unchanged VLM, on the teacher's own responses | full VLM; only the student trains | optional stage 2 | `+experiment=logit/gemma4_vit_l8_d384_e2b` |

Whatever the type, the student outputs the same thing: exactly one 768-d token per 48-px cell of
the processor output, in Gemma's order, at every budget of the declared input domain. It is a
drop-in vision tower for both E2B and E4B. Gemma 4's `embed_vision` normalises every token
(scale-free RMSNorm), so the language model sees only token directions. All losses and the shared
fidelity metrics (`fid_*`, also the checkpoint-selection criterion) are therefore invariant to the
per-token scale.

Each task is a class in `onboard_vlm/tasks/`, built TimeFM-style as `Task(hparams, **task_cfg)` from
`config/task/<type>.yaml`. Experiments are grouped by type (`config/experiment/{feature,embedding,logit}/`).
Details: [docs/design.md](docs/design.md#5-distillation-types-are-tasks).

## Real-time logging

Every stage is logged as it happens, to stdout and `<run dir>/run_rank0.log`. From the offline dry run:

```
[..09:26:10][rank0][onboard_vlm.run] ===> Run 'dryrun_feature' | task=FeatureDistillationTask | teacher=tiny | student=compact_vit | data=DistillDataModule
[..09:26:10][rank0][onboard_vlm.run] ===> Building task onboard_vlm.tasks.feature_distillation.FeatureDistillationTask
[..09:26:10][rank0][onboard_vlm.tasks] ===> Building student 'compact_vit'
[..09:26:11][rank0][onboard_vlm.tasks] <=== Building student 'compact_vit' done in 0.7s | peak RSS 0.8 GB
[..09:26:11][rank0][onboard_vlm.tasks]      criterion: 1*tok_mse[TokenFeatureLoss] + 1*tok_cos[TokenFeatureLoss] + 1*rel[RelationalTokenLoss] + ...
[..09:26:32][rank0][onboard_vlm.data.data_module] ===> Setting up datasets
[..09:26:32][rank0][onboard_vlm.data.data_module]      train[0] name=synthetic -> 64 samples
[..09:26:32][rank0][onboard_vlm.tasks] ===> Loading teacher tiny (load=vision)
[..09:26:32][rank0][onboard_vlm.train] ===> Training: FeatureDistillationTask
[..09:26:33][rank0][onboard_vlm.train] step 1/4 | epoch 0 | train/tok_cos=1.021 train/total=9.978 ... | lr 0.000855 | 4.4 img/s | elapsed 0:00:00 | ETA 0:00:02 | peak RSS 1.0 GB
[..09:26:33][rank0][onboard_vlm.train] ===> Validation (epoch 0, step 2)
[..09:26:35][rank0][onboard_vlm.run] ===> Exporting deployable student to outputs/students/dryrun_feature
```

With the real teacher, the weight download/load is its own stage:

```
===> Loading original Gemma 4 weights google/gemma-4-E2B-it@4adf157 (torch.bfloat16, attn=sdpa)
     loaded Gemma4ForConditionalGeneration: ... in N s
     teacher vision tower: Gemma4VisionModel: 167.36M params ...
```

## Layout

```
onboard_vlm/
  tasks/     base_task.py feature_distillation.py embedding_distillation.py logit_distillation.py
  teachers/  gemma4.py feature_cache.py            # original Gemma 4 via from_pretrained + offline cache
  students/  gemma4_vit.py timm_spatial.py vit.py multiplexed_vit.py io.py
  distill/   adapter.py losses.py fidelity.py      # shared, scale-invariant fidelity metrics
  data/      corpus.py datasets.py collator.py degrade.py hashing.py lmdb_io.py prompts.py samplers.py data_module.py
  vlm/       grafted.py                            # hot-swap vlm.model.vision_tower
  eval/      benchmarks.py runner.py metrics/      # official metrics, LLM judges, submissions
             lmms_eval_model.py probes.py probe_data.py stats.py   # E3 bridge, E2 probes, bootstrap / paired tests
  utils/     logging.py callbacks.py profiler.py tiny.py
config/      task/ criterion/ teacher/ student/ data_module/ trainer/ experiment/{feature,embedding,logit,sweeps,debug} benchmark/
make_datasets/
  download/*.sh  lmdb/make_*.py                    # benchmarks -> LMDB
  download/corpus/*.sh  corpus/*.py  build_corpus.sh   # general-vision corpus -> LMDB
  eval_sets/                                       # E1 / E3 evaluation image sets (fidelity + decontamination)
run_distill.py  run_generate_targets.py            # training; teacher responses for logit distillation
run_fidelity.py  run_probe.py  run_eval.py  run_lmms_eval.py  run_profile.py  run_cache_teacher.py
scripts/  eval_general.sh  aggregate_results.py    # E3 general suite; paper tables with CIs
scripts/  paths.sh  check_data.py  slurm/                       # data layout, status, batch jobs (submit_data_pipeline.sh)
make_datasets/prepare_all.sh                       # every dataset: download -> processed, in order
docs/  protocol.md  preliminary.md  shrink_study.md  setup_cscs.md  design.md  corpus.md  datasets.md  metrics.md
```

## Install

On CSCS Alps (or any cluster): [docs/setup_cscs.md](docs/setup_cscs.md) covers the clone, the conda
environment, the one-time access steps, and every dataset from download to processed format
(`make_datasets/prepare_all.sh`). Raw and processed data are kept apart (`scripts/paths.sh`).

```bash
conda env create -f environment.yml && conda activate onboardvlm
```

```bash
pip install torch torchvision && pip install -e ".[data,dev]"
```

## Running experiments (all YAML)

```bash
python -u run_distill.py +experiment=debug/dryrun_feature        # offline: tiny random Gemma 4, synthetic data, CPU
```

```bash
python -u run_distill.py -m +experiment=embedding/gemma4_vit_l8_d384_e2b seed=0,1,2          # main method, 3 seeds
```

```bash
python -u run_distill.py +experiment=embedding/gemma4_vit_l8_d384_e2b teacher=gemma4_e4b student.num_hidden_layers=6 student.hidden_size=512
```

```bash
python -u run_distill.py +experiment=feature/gemma4_vit_l8_d384_e2b                             # ablation
```

```bash
python -u run_generate_targets.py && python -u run_generate_targets.py split_name=val 'sources=${val_sources}'
```

```bash
python -u run_distill.py -m +experiment=logit/gemma4_vit_l8_d384_e2b seed=0,1,2               # stage 2 from the embedding export
```

```bash
python -u run_distill.py +experiment=baselines/projector_mobileclip2_mci2_e2b                   # same data and compute
```

```bash
python -u run_fidelity.py student_dir=outputs/students/embedding_gemma4_vit_l8_d384_e2b_s0 run_name=g4vit_s0   # E1
```

```bash
python -u run_distill.py -m +experiment=sweeps/cnn_zoo
```

```bash
python -u run_distill.py -m +experiment=sweeps/corpus_mixture
```

```bash
python run_eval.py -m benchmark=vizwiz_vqa,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema_subset,cvbench student_dir=outputs/students/embedding_gemma4_vit_l8_d384_e2b_s0 run_name=g4vit_s0
```

```bash
bash scripts/eval_general.sh g4vit_s0 outputs/students/embedding_gemma4_vit_l8_d384_e2b_s0      # E3 general suite (lmms-eval)
```

The last two commands evaluate an exported student on the target benchmarks and on the general
lmms-eval suite. `scripts/aggregate_results.py` turns all outputs into tables with bootstrap CIs and
paired tests against the teacher.

Training batches hold one budget and one processor shape each (no padding) and about
`tokens_per_batch` soft tokens. Every budget of `input_domain` is trained, with exact compute shares
(`train_budgets`, `train_budget_weights`). Validation runs per group at `val_budgets`; the test split
runs once at every budget.

Hydra merges dicts, so replace one with `~layer_map '+layer_map={1: 2}'`. Writing `layer_map={...}`
only adds keys.

### Students (`config/student/`)

Every family has one fully commented base config, and presets inherit from it. Any knob can be
overridden on the CLI. Profile below: 1080×1440 frame at 280 tokens (672×912 px, 266 tokens), from
`run_profile.py`.

| Family | Presets | Knobs | GMACs | Params |
|---|---|---|---|---|
| **Gemma 4 teacher** (reference) | | 16 L × 768, 12 heads | 539 | 167.4M |
| **`gemma4_vit`**: a smaller Gemma 4 encoder (the unchanged `Gemma4VisionModel` code, smaller config) | `gemma4_vit_l8_d768`, `_l12_d512`, `_l8_d512`, `_l8_d384` (default), `_l6_d256` | `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `head_dim`, `intermediate_size`, `position_embedding_size`, `use_clipped_linears` | 253 / 192 / 129 / 81 / 33 | 76.5 / 51.4 / 34.6 / 19.7 / 6.8M |
| **`timm_spatial`**: CNNs / hybrids / efficient ViTs on the exact 48-px grid | MobileNetV3/V4 (conv S/M/L, hybrid M/L), MobileNetV5-300M (Gemma 3n), EfficientNet B0/B2, EfficientNetV2 B0/B3/S, FastViT, MobileCLIP / MobileCLIP2 MCi0–3, RepViT, EfficientViT B/L/**M**, GhostNetV3, StarNet, TinyViT, MobileOne, HGNetV2, ConvNeXt-V2, EdgeNeXt, **MambaOut, SHViT, EfficientFormerV2, LeViT, MobileViTv2** | `model_name` (any timm id + pretrained tag), `alignment: input_rescale\|stem_stride3\|tap16_conv3\|resample`, `head`, `model_kwargs` | MNv4-M 4.7, StarNet-S2 3.0, EffNetV2-B3 8.7, MCi2 33, MambaOut-femto 6.3, SHViT-S2 7.9; resample models 0.7–2.3 | 3.6–35M |
| `compact_vit` / `coarse_patch_vit` / `multiplexed_vit` | `compact_vit_s`, `coarse_patch_vit_s`, `multiplexed_vit_s(_lowrank)` | depth, width, heads, patch 16/48, MiniViT sharing | 104 / 7.1 / 237 | 21.9 / 24.2 / 7.9M |

**Why not just add a linear / MLP layer at the end of a CNN?** A per-token head changes channels,
not the number or position of tokens. A stride-32 CNN gives 2.25× too many cells, misaligned with
Gemma's 48-px cells. Fixed-resolution models crash before the head is reached. Instead:

* `input_rescale` (default) resizes the input by r/48 so the *unmodified* pretrained backbone lands
  exactly on the 48-px grid. This is the head-at-the-end idea, made exact.
* `resample` adds a learned position-query attention head for the fixed-resolution families.

All previously excluded families now work. Details and the shift-equivariance proof are in
[docs/design.md](docs/design.md#putting-a-cnn-on-gemmas-token-grid).

### Data

* **Benchmarks** (download → LMDB): [docs/datasets.md](docs/datasets.md). Official metrics:
  [docs/metrics.md](docs/metrics.md).
* **OVLM-GV, the general-vision distillation corpus** (`data_module=general_vision`): about 18M raw
  images, filtered, benchmark-decontaminated and de-duplicated. The exact counts are written to the
  generated corpus card. Sources:
  * web: CC12M;
  * objects: ImageNet-1k (ImageNet-21k optional);
  * scenes: Open Images V7 and SA-1B;
  * text-rich: PixMo-Docs, SynthDoG-en and The Cauldron documents / charts / UI;
  * egocentric: Ego4D and EPIC-KITCHENS-100 frames;
  * the target train splits.

  Each source is preprocessed once into its own LMDB dataset. Mixture weights, groups and the
  train / val / test split are set in YAML and applied by the loader, so every ablation is a CLI
  override with no rebuild. Benchmarks keep their official splits. Sources, licences, download, preprocessing,
  LMDB format, decontamination, de-duplication and the loader-side train / val / test split are in
  [docs/corpus.md](docs/corpus.md). Build it with:

```bash
RAW_ROOT=/data/raw LMDB_ROOT=/data/lmdb CORPUS_ROOT=/data/corpus WORKERS=32 bash make_datasets/build_corpus.sh
```

## Corrections to the original brief

| Brief | Reality (verified) |
|---|---|
| Processor pads images to multiples of 48 | It *resizes* (and upsamples small images) to multiples of 48 and pads only the patch sequence. Our geometry is bit-exact to the official processor. |
| Collator emits normalised images | Gemma uses no mean/std; students normalise internally. |
| EgoBlind / EgoTextVQA string metrics | Both use official GPT-4o-mini judges (prompts reproduced verbatim); see docs/metrics.md. |
| 12B "Unified" is encoder-free | Correct: it has no vision tower and is rejected, like every non-E2B/E4B id. |
| E2B and E4B share a tokenizer | Almost: 1 of 262,144 vocabulary entries and 14 BPE merges differ. Cross-model logit KD is flagged as approximate. |

## Tests

```bash
pytest
```

255 tests, CPU-only and offline (tiny random Gemma 4). They cover:

* geometry parity with `Gemma4ImageProcessor`;
* teacher/student shape parity for every family;
* exact 48-px alignment for every CNN alignment mode, including one-token shift equivariance and a
  stride-32 negative control;
* the `gemma4_vit` student reproducing the teacher exactly at full size;
* all three tasks training, with the logit task leaving the VLM byte-identical;
* real-time logging;
* the corpus pipeline: filters, decontamination by content / id / video, global de-duplication,
  content-hashed train / val / test splits, leak checks, per-group metrics and the mixture sampler;
* the protocol:
  * exact input-domain enumeration and student coverage;
  * shape-exact, token-balanced batches with exact budget shares across DDP ranks;
  * scale-invariant fidelity metrics;
  * the truncated-teacher and frozen-projector baselines;
  * teacher-generated logit targets;
  * E1 end to end;
  * bootstrap, paired and McNemar statistics, and table aggregation;
  * blocklisting of the evaluation image sets;
* every YAML preset and experiment, plus grafted `generate()` for images and video.
