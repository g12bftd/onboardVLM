# Preliminary results on a partial corpus

What can be run before every licence and download is in place, and how, without compromising the
protocol of [protocol.md](protocol.md). Everything here is **provisional**: final numbers are re-run on
the complete corpus with the same commands minus `allow_missing_sources=true`.

## 1. What a partial run changes, and what it does not

| unchanged | changed |
|---|---|
| decontamination: the blocklist covers every benchmark and evaluation set built; `corpus_finalize` re-applies it to every image, so benchmarks added later need only `corpus_finalize` again (no source is rebuilt) | sources not built (`--skip`) are left out of the corpus (`corpus_manifest.json`: `"preliminary": true`, `missing_sources`) |
| content-hashed train / val / test splits (adding sources later never moves an image between splits) | the mixture renormalises over the sources present (`allow_missing_sources=true`; every left-out entry is logged as `PRELIMINARY`) |
| selection on val only, test once, equal compute, the statistics of §5 | E1 / E3 skip the sets that are not built (`skipped_sets.json`) |

## 2. Preliminary corpus

Built with `--skip sa1b,ego4d,epic_kitchens,egoblind,egotextvqa` (section 6):

| group | sources present | mixture weight (paper) | renormalised |
|---|---|---:|---:|
| web | CC12M | 0.33 | 0.44 |
| objects | ImageNet-1k train | 0.12 | 0.16 |
| scenes | Open Images (SA-1B missing) | 0.10 (+0.08) | 0.13 |
| text_rich | PixMo-Docs, SynthDoG-en, The Cauldron (13 subsets) | 0.17 | 0.23 |
| target | VizWiz train, TextVQA train (EgoBlind train missing) | 0.03 | 0.04 |
| egocentric | **none** (Ego4D, EPIC-KITCHENS missing) | 0.17 | 0 |

A quarter of the paper mixture is missing, including the whole egocentric group, the domain of the
target device. Preliminary results therefore measure general-vision distillation; egocentric fidelity
(E1 on EgoSchema frames is still available) is expected to improve once Ego4D / EPIC-KITCHENS are added.

Benchmarks built: VizWiz, TextVQA, WearVQA, CV-Bench, EgoSchema, EgoTextVQA-Indoor (from the EgoSchema
clips). Not yet: EgoBlind (one test archive missing), EgoTextVQA-Outdoor (not downloaded).

## 3. Students

Every student is defined, unit-tested (build, forward, input-domain coverage) and none is trained yet.
Parameters of the vision encoder that replaces Gemma 4's (16 layers x 768):

| family | config | params |
|---|---|---:|
| Gemma-architecture ViT (main) | `gemma4_vit_l8_d384` | 19.7M |
| | `gemma4_vit_l6_d256` / `_l8_d512` / `_l12_d512` / `_l8_d768` | 6.8M / 34.6M / 51.4M / 76.5M |
| efficient ViTs | `compact_vit_s` / `coarse_patch_vit_s` / `multiplexed_vit_s` | 21.9M / 24.2M / 7.9M |
| pretrained timm backbones + grid head | `timm_mobileclip2_mci2` / `_mobilenetv4_conv_m` / `_efficientnetv2_b3` / `_fastvit_sa12` / `_repvit_m1_5` / `_efficientvit_b1` | 35.4M / 7.9M / 12.6M / 10.8M / 14.0M / 4.8M |
| | `timm_mobilenetv5_300m_gemma3n` (Gemma 3n's encoder) | 279.9M |

## 4. Experiments, in order

Every comparison changes one factor; data, budgets, objective weights, compute (images and soft tokens
seen, recorded in `train_summary.json`) and seeds are otherwise equal.

**Phase A: no training** (data ready as soon as the benchmark LMDBs exist; the held-out E1 / E2 sets
are ready now).

| id | question | runs |
|---|---|---|
| A1 | floor: how faithful is the teacher's own first *k* layers, untrained? | E1 for `gemma4_truncated` k = 4, 8, 12 |
| A2 | representation ceilings: teacher and off-the-shelf small encoders | E2 for the teacher, MobileCLIP2-MCi2, MobileNetV4-M, FastViT-SA12, MobileNetV5-300M |
| A3 | cost: params, MACs, activation memory per budget | `run_profile.py` (no data) |

**Phase B: training** (after `corpus_finalize`). Preliminary compute budget: `data_module=general_vision_lite
trainer.max_epochs=10` (5M image samples per run); B0 measures throughput first.

| id | question | runs (x 3 seeds) | compared with |
|---|---|---|---|
| B0 | throughput and stability | main config, `trainer.max_steps=300`, 1 seed | — |
| B1 | main method | `embedding/gemma4_vit_l8_d384_e2b` | — |
| B2 | does the objective matter? | `feature/gemma4_vit_l8_d384_e2b` | B1 |
| B3 | is training needed beyond the teacher's own layers? | `baselines/truncated_teacher_l8_e2b` | B1, A1 |
| B4 | distil the encoder, or adapt an existing efficient one? | `baselines/projector_mobileclip2_mci2_e2b` (frozen backbone) | B1 |

Each trained run is evaluated with E1 (all sets present, all budgets) and E2.

**Phase C: downstream (E3)**, the student grafted into Gemma 4 E2B, rule-based scores only: VizWiz val
and TextVQA val (VQA accuracy), CV-Bench and the EgoSchema subset (multiple-choice accuracy), and the
lmms-eval suite with `DETERMINISTIC=1` (drops the two GPT-judged tasks). The teacher at 70 tokens is the
"fewer tokens" baseline.

## 5. Statistics (fixed before the results)

* **Primary endpoint** of every comparison: per-image LM-input-space cosine (`lm_cos`) at 280 tokens on each
  of the 7 held-out E1 sets (COCO, ADE20K, ImageNet val, DocVQA, InfoVQA, ChartQA, RealWorldQA).
  Secondary: the other budgets, relative error, CKA, the corpus and target sets, E2, E3.
* **Unit and pairing**: the evaluation item (image for E1, question for E3); every model is evaluated on the
  same items, so comparisons are paired.
* **Uncertainty**: 3 training seeds per configuration. Differences are estimated with a paired bootstrap that
  resamples items *and* each model's seeds (`stats.hierarchical_paired_bootstrap`, 10k resamples), so a
  difference must hold across training runs, not only across images. Also reported: seed mean +- std, and a
  95% bootstrap CI per model.
* **Multiplicity**: Holm correction over the 7 held-out sets within each comparison (and over benchmarks for E3);
  binary E3 outcomes also get the exact McNemar test.
* **Reporting**: effect sizes with CIs for every comparison, including null results; no tuning on test sets.

## 6. Commands (login node, in the repo, `(onboardvlm)` active)

```bash
cd /users/$USER/repos/onboardVLM && git pull && source scripts/paths.sh
```

Data: benchmark LMDBs, blocklist, corpus sources and a preliminary finalize (re-run later without `--skip`):

```bash
bash scripts/slurm/submit_data_pipeline.sh --account <project> --skip sa1b,ego4d,epic_kitchens,egoblind,egotextvqa
```

Phase A (`scripts/slurm/run.sbatch` runs one entry point on a node; evaluation uses one GPU):

```bash
for k in 4 8 12; do sbatch --account=<project> --job-name=e1-trunc$k --time=12:00:00 scripts/slurm/run.sbatch run_fidelity.py run_name=truncated_l$k "student={name: gemma4_truncated, num_layers: $k}" allow_missing_sources=true; done
```
```bash
sbatch --account=<project> --job-name=e2-teacher --time=12:00:00 scripts/slurm/run.sbatch run_probe.py run_name=teacher_e2b encoder.teacher=true
```
```bash
for m in fastvit_mci2.apple_mclip2_dfndr2b mobilenetv4_conv_medium.e500_r256_in1k fastvit_sa12.apple_dist_in1k mobilenetv5_300m.gemma3n; do sbatch --account=<project> --job-name=e2-$m --time=12:00:00 scripts/slurm/run.sbatch run_probe.py run_name=$m encoder.kind=timm encoder.model_name=$m; done
```
```bash
sbatch --account=<project> --job-name=profile --time=02:00:00 scripts/slurm/run.sbatch run_profile.py
```

Phase B, once `corpus-finalize` has finished (B0 first; then three seeds of each experiment):

```bash
P="data_module=general_vision_lite allow_missing_sources=true trainer.max_epochs=10"
```
```bash
sbatch --account=<project> --job-name=b0 --time=02:00:00 scripts/slurm/run.sbatch run_distill.py +experiment=embedding/gemma4_vit_l8_d384_e2b $P experiment_name=prelim_b0 trainer.max_steps=300
```
```bash
for s in 0 1 2; do for e in embedding/gemma4_vit_l8_d384_e2b feature/gemma4_vit_l8_d384_e2b baselines/truncated_teacher_l8_e2b baselines/projector_mobileclip2_mci2_e2b; do n=prelim_${e//\//_}; sbatch --account=<project> --job-name=$n-s$s scripts/slurm/run.sbatch run_distill.py +experiment=$e $P experiment_name=$n seed=$s; done; done
```

E1 / E2 of a trained run (`<run_id>` = `<experiment_name>_s<seed>`; the export is `$ONBOARD_VLM_OUT/students/<run_id>`):

```bash
sbatch --account=<project> --job-name=e1-<run_id> --time=12:00:00 scripts/slurm/run.sbatch run_fidelity.py run_name=<run_id> student_dir=$ONBOARD_VLM_OUT/students/<run_id> allow_missing_sources=true
```
```bash
sbatch --account=<project> --job-name=e2-<run_id> --time=12:00:00 scripts/slurm/run.sbatch run_probe.py run_name=<run_id> encoder.student_dir=$ONBOARD_VLM_OUT/students/<run_id>
```

Tables (several comma-separated directories = seeds of one model; `--reference` is what every model is
compared with):

```bash
O=$ONBOARD_VLM_OUT/fidelity; E=prelim_embedding_gemma4_vit_l8_d384_e2b; F=prelim_feature_gemma4_vit_l8_d384_e2b
python scripts/aggregate_results.py --out $ONBOARD_VLM_OUT/tables/prelim_b2 --reference embedding \
  --fidelity embedding=$O/${E}_s0,$O/${E}_s1,$O/${E}_s2 --fidelity feature=$O/${F}_s0,$O/${F}_s1,$O/${F}_s2 \
  --fidelity truncated_l8=$O/truncated_l8
```
