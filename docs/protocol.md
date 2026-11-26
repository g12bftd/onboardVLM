# Experimental protocol

This is the protocol the code implements, written as it would appear in a paper's experimental
section and appendix.

## 1. Claims and the evidence for each

| claim | evidence | where |
|---|---|---|
| the student reproduces the teacher's vision tokens | E1 fidelity, in the language model's input space, per budget, aspect ratio and image group | `run_fidelity.py` |
| the student keeps the teacher's general visual representation | E2 frozen-feature probes (classification, segmentation, depth) against prior efficient encoders | `run_probe.py` |
| the VLM with the student keeps its abilities | E3 downstream: target benchmarks plus the lmms-eval general suite, teacher vs student in the unchanged Gemma 4 | `run_eval.py`, `run_lmms_eval.py` |
| the student is what makes this possible, not other choices | same-LM baselines at equal training compute (§5) | `config/experiment/baselines/` |
| it is efficient on device | params, MACs per budget, latency, memory | `run_profile.py` |

Every number comes with a 95% bootstrap CI. Paired comparisons with the teacher are Holm-corrected,
and main results report the mean ± std over three training seeds (§6).

## 2. The student contract

Gemma 4's processor resizes each image to H×W, multiples of 48 px, with (H/48)(W/48) ≤ budget,
where the budget is one of {70, 140, 280, 560, 1120}. The vision tower emits one 768-d token per
48-px cell, in row-major order, and `embed_vision` maps the tokens to the language model.

**The student is a drop-in replacement for the vision tower.** Same input, i.e. the processor
output with bit-exact geometry. Same output: exactly (H/48)(W/48) tokens of width 768, in the same
order, token *k* describing cell *k*. All three distillation types train and export this same
artifact. It is used unchanged by E2B and E4B, which share the tower.

**The language model sees only each token's direction.** `embed_vision(z) = W · z / rms(z)`, with a
scale-free RMSNorm. Losses and metrics must therefore be invariant to a positive per-token
rescaling; plain MSE on vision tokens is not, and is dominated by the teacher's large-norm tokens.

| distillation type (`task=`) | the student output is compared … | loss | role in the paper |
|---|---|---|---|
| `embedding_distillation` (default) | after the checkpoint's frozen `embed_vision`: exactly the LM input | MSE + cosine in LM-input space | **main method**: exact and task-agnostic; errors count by how much they move the LM input (metric WᵀW) |
| `feature_distillation` | in the 768-d vision space | RMS-normalised MSE + cosine, per-image relational, optional intermediate layers | ablation: the embedding loss with W replaced by the identity |
| `logit_distillation` | through the unchanged VLM: next-token distributions | KL on the teacher's *own* responses | optional stage 2, initialised from the embedding export |

## 3. Input domain

`input_domain: {budgets: [70, 140, 280, 560, 1120], max_aspect: 4.0}` declares the set of inputs a
student is trained on and claimed on: every processor output for source aspect ratios in [1/4, 4].

* The set is enumerated exactly (`geometry.processor_shapes`). The processor's size depends only on
  the aspect ratio, so the code evaluates the reference implementation at all its breakpoints. That
  gives 311 shapes and at most 198 patches per axis. The enumeration was checked against 567,541
  random and adversarial inputs.
* Students with learned position tables must cover the domain; this is checked when they are built.
  The 256-row default covers 4:1; an 8:1 domain would need 282.
* At evaluation, inputs outside the domain are excluded and counted (E1), or counted and logged
  (grafted VLM, E3). Glasses cameras (4:3, 16:9, either orientation) are well inside the domain.

## 4. Training

**Data.** OVLM-GV ([corpus.md](corpus.md)): 6 groups and ~18M raw images, decontaminated against
every evaluation image (§7) and de-duplicated. Train/val/test splits are a seeded content hash
assigned by the loader (`corpus_split`: 0.2% val, 0.2% test).

**Batches** (`BucketedBatchSampler`):

* One budget and one processor shape per batch, so there is never any padding. Students see exactly
  what they see at batch size 1 on the device, and CNN and ViT students are trained identically.
* About `tokens_per_batch` = 8,960 soft tokens per batch: ~143 images at 70 tokens, 32 at 280, 8 at
  1120.
* Each budget gets an exact share of batches (`train_budget_weights`; uniform by default, i.e. equal
  compute per budget). The share is allocated, not sampled, and does not depend on when the epoch
  ends.
* Aspect ratio: 50% camera aspects, 50% log-uniform over the domain. Each image is randomly cropped
  to the batch's aspect ratio (area 35–100% of the largest such crop) and resized with the
  processor's resize. Teacher and student always see the same pixels.
* Every rank builds the same plan and takes its share; all samplers handle DDP themselves.

**Optimisation.** AdamW, linear warmup, cosine decay. Configs are in `config/task/*.yaml`.

**Compute accounting.** `train_summary.json` records:

* images and soft tokens seen, per budget;
* optimizer steps;
* wall clock and device-hours.

Runs that are compared must have equal soft tokens seen (`tokens_per_batch` × steps).

**Model selection.** Only on the validation split: the corpus val split per group, plus the
official VizWiz val, at budgets {70, 280, 1120}. All three types select on the same metric:
`val/fid_lm_cos`, the image-weighted LM-input-space cosine. Test splits are evaluated once, after
training (`final_test`).

**Seeds.** `run_id = <experiment>_s<seed>` keeps runs apart. Main results use three seeds:

```bash
python -u run_distill.py -m +experiment=embedding/gemma4_vit_l8_d384_e2b seed=0,1,2
```

Logit distillation of seed *s* starts from the embedding export of seed *s*.

**Logit-distillation targets** (`run_generate_targets.py`):

* the original VLM answers greedily at 280 tokens;
* prompts are six image-agnostic templates over corpus train images (caption, detailed caption,
  OCR, spatial layout, assistive description, salient content), plus the VizWiz and TextVQA train
  questions;
* the cache stores the exact prompt, the response and its provenance (revision, budget, generation
  settings);
* validation responses come from corpus val images;
* distilling on human reference answers is the ablation (`logit/*_human_answers`).

## 5. Baselines (same language model, same data and compute)

| baseline | what it isolates | how |
|---|---|---|
| teacher at a lower budget | is a small student better than simply using fewer tokens? Compare at equal MACs (`run_profile.py` gives MACs per budget) | `run_eval.py image_token_budget=70`; `scripts/eval_general.sh teacher_b70 "" google/gemma-4-E2B-it 70` |
| truncated teacher (first *k* of 16 layers, own pooler, no training) | does training matter, or is depth alone enough? | `run_fidelity.py 'student={name: gemma4_truncated, num_layers: 8}'`; `run_eval.py 'student={...}'` |
| truncated teacher + distillation | the strongest initialisation-based baseline | `+experiment=baselines/truncated_teacher_l8_e2b` |
| off-the-shelf efficient encoder + trained projector (backbone frozen) | is distilling the encoder better than adapting an existing small encoder (LLaVA-style)? | `+experiment=baselines/projector_{mobileclip2_mci2, mobilenetv4_conv_m, mobilenetv5_300m_gemma3n, fastvit_sa12}_e2b` |
| published small VLMs (Gemma 3n, FastVLM, SmolVLM2, InternVL3-1B/2B, Qwen2.5-VL-3B) | context only: their language models differ | published or lmms-eval numbers |

## 6. Evaluation

**E1 — fidelity** (`run_fidelity.py`, `config/fidelity.yaml`).

* Every evaluation set at every budget of the domain, in deployment precision (bf16), with
  shape-exact batches.
* Sets:
  * in-distribution: the corpus test split per group;
  * held out: COCO val2017, ADE20K val, ImageNet-1k val, DocVQA val, InfographicVQA val,
    ChartQA test, RealWorldQA;
  * target: VizWiz val, TextVQA val, WearVQA, CV-Bench, EgoSchema frames.
* Metrics (`onboard_vlm/distill/fidelity.py`, identical to the training-time validation metrics):
  * primary: LM-input-space cosine and relative error;
  * also: vision-space cosine, per-image CKA of the RMS-normalised token sets, and the 5th-percentile
    token cosine.
* Outputs: per-image records (for fidelity ↔ accuracy correlation) plus bootstrap summaries per set,
  budget and aspect bin.

**E2 — representation probes** (`run_probe.py`). One protocol for students, the teacher, timm models
and HF vision towers:

* ImageNet-1k (100 images per class for training): weighted k-NN (k=20, T=0.07) and L-BFGS logistic
  regression, with L2 chosen on a held-out 10%;
* ADE20K: linear segmentation probe (mIoU);
* NYUv2 (official 795/654 split): linear log-depth probe (RMSE, AbsRel, δ1).

Gemma-family encoders are probed at the 280-token budget; external encoders at their native
resolution.

**E3 — downstream**, with the student grafted into the unchanged Gemma 4.

* Target benchmarks (`run_eval.py`, official metrics and judges): VizWiz-VQA, TextVQA, WearVQA,
  EgoBlind, EgoTextVQA (outdoor high/low, indoor), EgoSchema, CV-Bench.
* General suite (`scripts/eval_general.sh`, lmms-eval 0.7.4's own `gemma4` model subclassed, so
  prompts and scoring match published Gemma 4 numbers):

  | area | benchmarks |
  |---|---|
  | perception and hallucination | POPE, MMVP, HallusionBench |
  | general | MMBench-EN, MMStar, SEED-Bench, RealWorldQA |
  | OCR and documents | OCRBench, DocVQA, ChartQA, InfoVQA, AI2D |
  | spatial | BLINK, CV-Bench |
  | control | MMMU |
  | video, at 70 tokens per frame | Video-MME, MVBench |

* Every student is evaluated in **both** E2B and E4B.

**Efficiency** (`run_profile.py`): params, MACs and peak activation memory at every budget for a
1440×1080 frame; latency on the target device.

## 7. Statistics and data hygiene

* `onboard_vlm/eval/stats.py` and `scripts/aggregate_results.py` compute:
  * a percentile-bootstrap 95% CI (10k resamples) over evaluation items for every score;
  * paired comparisons on the same items (paired bootstrap of the mean difference; exact McNemar for
    binary outcomes), Holm-corrected across benchmarks. With several training seeds the bootstrap
    resamples each model's seeds as well as the items (`hierarchical_paired_bootstrap`), so a difference
    must hold across training runs;
  * retention = student / teacher;
  * seed mean ± std.
* Per-item scores come from `scored.jsonl` (`run_eval.py`), lmms-eval `--log_samples` files and
  `per_image.jsonl` (E1).
* **Decontamination.** `make_eval_blocklist.py` fingerprints (SHA-1 + dHash) every non-train image
  and frame of the eight target benchmarks, every E1 and E2 set, and every lmms-eval E3 image set.
  The E3 images are built from the exact HF repos lmms-eval evaluates (`make_datasets/eval_sets/`).
  It also blocks the Open Images ids behind TextVQA and the Ego4D videos behind EgoSchema /
  EgoTextVQA-Indoor. Every corpus image is checked against this list when its source is built, and again
  by `corpus_finalize` (so benchmarks added later only need `corpus_finalize`, not a rebuild). Video-MME and
  MVBench videos are not fingerprinted: their sources are not in the corpus.
* **Leakage guards.**
  * One corpus split definition is used everywhere (training, target generation, E1), and a test
    checks that the configs agree.
  * The data module rejects conflicting split definitions, and val/test entries that read the
    wrong split.
  * Logit-distillation targets are generated only from corpus train images.
  * Probes use official train/eval splits.

## 8. Ablations (one config or CLI override each)

| question | how |
|---|---|
| distillation type | `+experiment={embedding,feature,logit}/...` (same student, data, compute) |
| which training budgets are needed | `+experiment=sweeps/token_budget` (all evaluated on every budget) |
| corpus composition | `+experiment=sweeps/corpus_mixture` (leave one group out) |
| student family / size | `+experiment=sweeps/gemma4_vit_depth_width`, `sweeps/cnn_zoo` |
| CNN input resolution | `student.alignment=stem_stride3` (full-resolution input) vs `input_rescale` (2/3 scale) |
| logit-distillation targets | `logit/gemma4_vit_l8_d384_e2b` vs `..._human_answers` |
| E2B vs E4B embedding space | `embedding/gemma4_vit_l8_d384_{e2b,e4b}`, each evaluated in both VLMs |
| split variance | `corpus_split.split_seed=1,2` |
| ImageNet-1k vs ImageNet-21k as the objects source | `objects_source=imagenet21k` (weights in `general_vision.yaml`) |

## 9. Run order

1. Benchmarks → LMDB: `make_datasets/download/download_all.sh`, then `make_datasets/build_all_lmdb.sh`.
2. Evaluation image sets: `python make_datasets/eval_sets/build_eval_sets.py --raw_root $RAW_ROOT/eval_sets --eval_root $EVAL_ROOT`.
3. Corpus: `make_datasets/download/corpus/download_corpus_all.sh`, then `bash make_datasets/build_corpus.sh` (blocklist → sources → finalize).
4. Main method, three seeds: `python -u run_distill.py -m +experiment=embedding/gemma4_vit_l8_d384_e2b seed=0,1,2`.
5. Optional stage 2: `python -u run_generate_targets.py` (train), then `split_name=val 'sources=${val_sources}'`, then `python -u run_distill.py -m +experiment=logit/gemma4_vit_l8_d384_e2b seed=0,1,2`.
6. Baselines and ablations (§5, §8), at equal compute.
7. Evaluation:
   * E1: `run_fidelity.py`;
   * E2: `run_probe.py`;
   * E3: `run_eval.py -m benchmark=...` and `scripts/eval_general.sh`, for the teacher and every student, in E2B and E4B.
8. Tables: `python scripts/aggregate_results.py --reference teacher ...`.

## 10. Limitations and what is not verified here

* Nothing has been run at scale in this repository: no real corpus, no real-weight training, no
  benchmark numbers. The pipeline is covered by 255 offline tests and CPU dry runs with a tiny
  random Gemma 4.
* The lmms-eval bridge is written against lmms-eval 0.7.4's `gemma4` model and registry API, read
  from its source. It has not been run end to end.
* The E1 dataset repos and parquet columns were checked on the Hugging Face Hub. The COCO and
  ADE20K archive URLs are the official ones but were not downloaded here.
* lmms-eval per-sample scores for paired tests are parsed heuristically (`aggregate_results.py`).
  Check them against lmms-eval's aggregate numbers.
* The E2 probes follow standard linear-probe protocols, but absolute numbers are not directly
  comparable to papers that use full ImageNet training sets or other resolutions. Compare encoders
  only within this protocol.
