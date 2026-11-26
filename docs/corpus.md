# OVLM-GV: the general-vision distillation corpus

Feature distillation is label-free: the student learns the teacher's function only on the images
it is shown. What the student keeps of the Gemma 4 encoder's general ability is therefore decided by
the image distribution of the corpus. OVLM-GV is built to cover that distribution along seven axes,
using only public, versioned sources, with every preprocessing step scripted and logged. It is
general-purpose by design: the goal is to keep Gemma 4's general vision-language abilities, so the
default mixture has no application domain and no benchmark images. Egocentric frames and benchmark
train images are an opt-in ablation (`data_module=general_vision_plus`, weights in its YAML).

| group | source (directory under `$CORPUS_ROOT`) | covers | images (raw) | licence / access | weight | precedent |
|---|---|---|---:|---|---:|---|
| web | CC12M, pixparse snapshot (`cc12m`) | open-vocabulary web photos, illustrations, products | 10,968,539 | CC12M terms; HF, open | 0.41 | MobileCLIP (DataCompDR), CLIP-KD |
| objects | ImageNet-1k train (`imagenet1k`; `objects_source`) | 1,000 object categories, object-centric photos | 1,281,167 | ImageNet terms (non-commercial research); HF, gated (auto-approved) | 0.15 | DeiT, TinyViT (ImageNet distillation) |
| scenes | Open Images V7, box-annotated train subset (`openimages`) | cluttered multi-object scenes, spatial layout | 1,743,042 | images CC BY 2.0, annotations CC BY 4.0; S3, open | 0.125 | |
| scenes | SA-1B, first 100 of 1,000 tars (`sa1b`) | high-resolution dense scenes | ~1.1M | SA-1B Research License; form | 0.10 | EfficientSAM / MobileSAM |
| text-rich | PixMo-Docs (`text_rich/pixmo_docs`) | charts, tables, diagrams, documents | 255,261 | ODC-BY 1.0; HF, open | 0.075 | Molmo |
| text-rich | SynthDoG-en (`text_rich/synthdog_en`) | dense rendered text | 500,000 | MIT; HF, open | 0.04 | Donut |
| text-rich | The Cauldron, 13 document / chart / scene-text / UI subsets (`text_rich/cauldron/*`) | real documents, infographics, scene text, screenshots | ~0.4M (4 synthetic subsets capped at 50k) | per subset; HF, open | 0.10 | Idefics2 |
| egocentric | Ego4D v2 `video_540ss`, ~1,000 h scenario-stratified, 0.5 fps (`egocentric/ego4d`) | head-mounted viewpoint, hands, motion blur, indoor/outdoor daily life | ≤1.8M frames | Ego4D licence; signed agreement + AWS keys | ablation only | |
| egocentric | EPIC-KITCHENS-100, 0.5 fps (`egocentric/epic_kitchens`) | egocentric manipulation, kitchens | ~0.18M frames | CC BY-NC 4.0; open | ablation only | |
| target | VizWiz, TextVQA, EgoBlind **train** splits (`target_train`) | blind-user photos, scene text, smart-glasses video | ~45K | per benchmark | ablation only | |

The weights are the sampling probabilities in `config/data_module/general_vision.yaml`
(`data_module.mixture`). After filtering, decontamination and de-duplication, the exact counts are
written to `$CORPUS_ROOT/CORPUS_CARD.md` and `corpus_manifest.json`. Report those numbers in the
paper, not the raw ones above.

## Why ImageNet-1k rather than ImageNet-21k

The objects group adds clean, object-centric photos with fine-grained categories (breeds, species,
artefacts) to the web, scene and text data. ImageNet-1k is the default
(`objects_source: imagenet1k`):

* **No extra download.** The E2 probes need the same 154 GB of ImageNet-1k train tars anyway;
  ImageNet-21k would add 1.19 TB to download and ~1 TB to store.
* **It is a subset of ImageNet-21k.** The fall11 release contains the ImageNet-1k train images, so
  what changes is breadth: 1k instead of 21k categories, and 1.28M instead of 13.7M images. CC12M
  (10.9M open-vocabulary web images) already supplies much of the long tail.
* **Standard precedent.** Encoder distillation on ImageNet-1k is the DeiT / TinyViT setting.
* **No leakage.** ImageNet-1k val is an evaluation set (E1, E2) and is blocklisted from the corpus;
  only train images enter it.

Its weight is lower than the 21k variant's (0.15 instead of 0.20, with CC12M at 0.41). That keeps
repetition at ~6 passes, comparable to the other components. Whether the objects group matters at
all is measured by the leave-one-group-out ablation (`sweeps/corpus_mixture`). Switching to
ImageNet-21k is one setting: `objects_source=imagenet21k` (download and build it first; the weights
for that variant are in `general_vision.yaml`).

## Why this corpus is defensible

* **Coverage, not just scale.** Each group exists because a general capability of the teacher, as
  the Cambrian-1 categories probe it ([protocol.md](protocol.md), E3), needs it:
  * web and object-centric images: recognition and world knowledge (General, Knowledge: MME, SEED,
    GQA, MMMU);
  * multi-object and high-resolution scenes: counting, spatial relations, depth order and detail at
    large token budgets (Vision-Centric: CV-Bench, MMVP, RealWorldQA; POPE);
  * text-rich images: documents, charts, diagrams and scene text (OCR & Chart: DocVQA, ChartQA,
    OCRBench, TextVQA; AI2D).

  Each group can be removed in a leave-one-group-out ablation
  (`+experiment=sweeps/corpus_mixture`), so its contribution is measured, not assumed.
* **Every source has precedent in encoder distillation.**
  * ImageNet: DeiT (ImageNet-1k), TinyViT (ImageNet-1k / 21k).
  * CC12M / DataComp-style web data: MobileCLIP and RADIO.
  * SA-1B: EfficientSAM and MobileSAM.
  * Ego4D (ablation only): the standard egocentric pre-training source.
* **Reproducible.** All sources are fixed snapshots: wds tars, parquet files, S3 tars and versioned
  video sets. No URL lists are re-scraped. LAION is excluded: it was withdrawn after CSAM reports, and
  Re-LAION would be one more URL-scraped set. DataComp-12M is excluded because it ships URLs only, so
  link rot would make the corpus differ between labs. The finalized training set is identified by a
  SHA-256 digest over the sorted SHA-1 of every training image (`train_sha1_digest` in the manifest).
* **Decontaminated against every evaluation set**, by content and by identity (see below), before
  de-duplication.
* **Licences are tracked per source.** The corpus is for research. ImageNet, SA-1B, Ego4D and
  EPIC-KITCHENS are non-commercial / research-only, so a model distilled on the full mixture inherits
  those terms. The leave-one-group-out configs let you report a commercially cleaner variant if needed.

## Pipeline

```
download (make_datasets/download/corpus/*.sh) -> $RAW_ROOT
benchmark LMDBs (make_datasets/build_all_lmdb.sh) -> $LMDB_ROOT
make_datasets/build_corpus.sh
   1 blocklist  : fingerprints of every benchmark val/test image + frame, TextVQA ids, EgoSchema videos
   2 sources    : decode -> filter -> decontaminate -> (downscale) -> LMDB shards + fingerprint index
   3 finalize   : global exact + perceptual de-dup, dataset.json per source, manifest + corpus card
train            : data_module=general_vision (or general_vision_lite)
                   the loader draws train / val / test and the mixture from YAML (no rebuild)
```

**Design rule: the data files hold data, the YAML holds the experiment.** Each source is
preprocessed once into its own LMDB dataset. Everything an ablation might change lives in
`config/data_module/*.yaml` and is applied by the loader:

* which sources are used;
* their mixture weights and groups;
* the train / val / test split (fractions and seed);
* the val / test set sizes.

So every ablation is a CLI override, and the resolved values are logged and saved with the run.
Weights and splits are deliberately *not* written into the LMDB files: changing them would mean
rewriting terabytes, and the run config would no longer be the full record of the experiment.

### 0. Download

The scripts resume interrupted downloads. Set `RAW_ROOT` (default `./data/raw`).

```bash
export RAW_ROOT=/data/raw CORPUS_ROOT=/data/corpus LMDB_ROOT=/data/lmdb
```

```bash
bash make_datasets/download/corpus/download_text_rich.sh      # PixMo-Docs, SynthDoG-en, 13 Cauldron subsets
```

```bash
bash make_datasets/download/corpus/download_openimages.sh     # 16 x train_*.tar.gz from the CVDF S3 bucket (~550 GB)
```

```bash
bash make_datasets/download/corpus/download_cc12m.sh          # pixparse/cc12m-wds, 2,176 tars (~1.2 TB)
```

```bash
hf auth login && bash make_datasets/download/corpus/download_imagenet1k.sh    # accept the terms on the HF page first (~154 GB; also used by the E2 probes)
```

```bash
SA1B_LINKS=/path/to/sa1b_links.txt SA1B_TARS=100 bash make_datasets/download/corpus/download_sa1b.sh
```

```bash
bash make_datasets/download/corpus/download_epic_kitchens.sh  # official epic_downloader.py --videos
```

```bash
EGO4D_HOURS=1000 bash make_datasets/download/corpus/download_ego4d.sh   # needs the blocklist (step 1) first
```

Access steps:

| source | what you need first |
|---|---|
| ImageNet-1k | accept the terms on huggingface.co/datasets/timm/imagenet-1k-wds (approved automatically) |
| SA-1B | accept the licence on ai.meta.com/datasets/segment-anything-downloads and save the link file |
| Ego4D | sign the licence on ego4d-data.org, then `aws configure --profile ego4d` with the keys you receive |

Ego4D is selected per video. `select_ego4d_videos.py` reads `ego4d.json` and drops every video behind
EgoSchema / EgoTextVQA-Indoor. It then fills the hour budget round-robin over scenarios, so cooking
and cleaning cannot crowd out the long tail of activities. It writes the uid list and a per-scenario
hour summary.

### 1. Benchmark blocklist (decontamination)

```bash
bash make_datasets/build_corpus.sh blocklist
```

Label-free distillation on evaluation images is still transductive leakage. The blocklist holds:

* SHA-1 and 64-bit dHash of every val/test image and every stored frame of all eight benchmark LMDBs.
  Train splits stay usable.
* Open Images ids of the TextVQA val/test images. TextVQA images *are* Open Images images, so they
  are blocked by id as well as by content.
* the Ego4D `video_uid` of every EgoSchema clip. EgoTextVQA-Indoor reuses these clips. The whole
  source video is excluded, not just the 3-minute clip.

Other overlaps the content fingerprints catch:

* CV-Bench 2D images come from COCO and ADE20K.
* ST-VQA (in The Cauldron) contains VizWiz images.

### 2. Build each source

```bash
WORKERS=32 bash make_datasets/build_corpus.sh text_rich openimages sa1b imagenet1k cc12m ego4d epic_kitchens
```

Every source goes through the same code (`make_datasets/corpus/corpus_common.py`). Images come from
tars, parquet or folders (`make_image_corpus.py`) or from videos (`make_frame_corpus.py`). Per image:

| step | default | why |
|---|---|---|
| decode + EXIF transpose | | the orientation the camera / benchmark loader sees |
| short side ≥ `--min_side` | 224 | below Gemma's smallest useful crop |
| long/short ≤ `--max_aspect` | 4 | banners and strips carry little visual content |
| grey 32×32 std ≥ `--min_std` | 3 | blank / near-uniform images |
| benchmark check | exact SHA-1, dHash ≤ 4 bits, source ids | decontamination (dHash with < 8 or > 56 set bits is exempt: unrelated low-texture images collide) |
| downscale long side ≤ `--max_side`, JPEG q93 | 1536 | covers the 560-token budget at any aspect and the 1120-token budget up to ~4:3; untouched otherwise |
| video frames: every 1/`--fps` s, skip if dHash ≤ `--still_bits` of the last kept frame | 0.5 fps, 6 bits | removes static-camera repeats |

Each worker writes its own shards plus `stats_<worker>.json` with counts per filter.
`--worker_id i --num_workers N` splits the input by tar / parquet / video.

**Storage format** (`onboard_vlm/data/corpus.py`), identical for every source:

```
$CORPUS_ROOT/<source>/
  shard_<worker>_<k>.lmdb         records under positional keys 00000000, 00000001, ...
  shard_<worker>_<k>.meta.json    {"format": "corpus-v1", "dataset", "num_samples"}
  shard_<worker>_<k>.index.npz    sizes [n, 2] | dhash uint64 [n] | sha1 S20 [n]
  stats_<worker>.json             build-time filter counts
  keep.npz                        de-duplication mask per shard (finalize)
  dataset.json                    provenance, licence, counts, duplicate statistics (finalize)
```

One source is one dataset directory. The shards inside it are an implementation detail: LMDB allows
only one writer per file, so a parallel build writes one shard per worker. The loader reads all
shards of a source (or the first `max_shards`) as a single dataset.

Each record keeps the encoded image bytes and its provenance (`sample_id = "<source>/<original
key>"`). The numpy index lets the loader start in seconds and use flat memory at 10^7+ images,
with no Python string per sample. The index is also all that de-duplication needs.

### 3. Finalize

```bash
bash make_datasets/build_corpus.sh finalize
```

`finalize_corpus.py` reads only the shard indices, so it takes minutes for the whole corpus. It
processes the sources in the order of `make_datasets/corpus/sources.yaml`: smaller, curated sources
come first and keep an image that also appears in a larger one. It applies:

1. **Exact de-duplication**: identical SHA-1, within and across sources.
2. **Perceptual de-duplication**: identical dHash, within and across sources. This catches re-encodes,
   resizes and re-hosted copies.

Outputs:

* `keep.npz` and `dataset.json` per source;
* `corpus_manifest.json`: counts per source; within- vs cross-source duplicates; build filter
  statistics; licences; the digest of all kept images;
* `CORPUS_CARD.md`: the same as a table for the paper's appendix, including the split sizes for the
  default split definition.

The paper configs set `require_finalized: true`, so a corpus that has not been finalized cannot be
used by accident.

### Train / val / test: assigned in the loader

`ImageCorpusDataset(split=..., val_fraction, test_fraction, split_seed)` assigns each image to a
split with a seeded hash of its fingerprint (`assign_split`):

* `u = splitmix64(dHash XOR seed)` in [0, 1); val if `u < val_fraction`, test if below
  `val_fraction + test_fraction`, otherwise train;
* the SHA-1 replaces the dHash for low-texture images.

What this guarantees, and what is tested:

* **No rebuild for a new split.** Fractions and seed are config values. Defaults are 0.2 % val and
  0.2 % test per source, which is ~22k images for CC12M and ~500 for PixMo-Docs. `limit` then takes
  an evenly spaced subset, so validation cost stays bounded.
* **Stable.** An image's split depends only on the image, the fractions and the seed, not on which
  other shards or sources are loaded:
  * the lite corpus is an exact subset of the full one, split by split;
  * adding a source never moves existing images between splits;
  * growing `val_fraction` only moves train images into val.
* **Duplicate-safe.** Copies with the same dHash always land in the same split, so even an
  un-finalized corpus cannot have a train image re-appear in val / test.
* **Hard to misconfigure.** `data_module.dataset_defaults.image_corpus` sets one split definition
  for every corpus entry. The data module refuses:
  * two entries reading the same source with different definitions;
  * a val entry that reads the train or test split.

**Benchmarks keep their official splits.** VizWiz, TextVQA, EgoBlind, etc. are stored as
`train.lmdb` / `val.lmdb` / `test.lmdb` exactly as released, and are never re-randomized. A random
re-split would leak official test images into training and make numbers incomparable with published
results. Only their train splits enter distillation; val / test are for evaluation and for the
decontamination blocklist.

## Using the corpus

```bash
python -u run_distill.py +experiment=feature/gemma4_vit_l8_d384_e2b            # data_module=general_vision
```

```bash
python -u run_distill.py +experiment=feature/gemma4_vit_l8_d384_e2b data_module=general_vision_lite
```

```bash
python -u run_distill.py -m +experiment=sweeps/corpus_mixture                   # leave-one-group-out
```

* **Sampling** is a fixed mixture (`BucketedBatchSampler`, [protocol.md](protocol.md#4-training)).
  Each epoch draws about `samples_per_epoch` (2M) images: every image's component is drawn by
  weight, then an image is drawn uniformly without replacement within that component. The plan is
  deterministic given seed + epoch, identical on every DDP rank, and each rank takes its share.
* **Repetition**. At start-up the data module logs, per component, the draws per epoch and how many
  passes over its images that means. With the default 30 × 2M draws:
  * web and scenes: 2–4 passes;
  * objects (ImageNet-1k): ~6 passes;
  * text-rich: 4–14 passes.

  Every pass is a new random crop at a sampled token budget. Following "a good teacher is patient
  and consistent" (Beyer et al., 2022), teacher and student always see the same augmented view,
  also under mixup (`data_module.collator.mixup_alpha`, [methods.md](methods.md)).
* **Validation** (`val`): one dataloader per (group, budget) for the groups web, objects, scenes and
  text-rich (corpus val split only), at `val_budgets` (70, 280, 1120).
  Images are batched by exact processor shape. Metrics are logged as `val/<group>/b<budget>/fid_lm_cos`,
  `val/<group>/...`, `val/b<budget>/...` and overall `val/...` (image-weighted). Checkpoints are
  selected on `val/fid_lm_cos` (docs/protocol.md).
* **Test** (`test`): the corpus test split per group, up to 5,000 images each, at every budget. It
  runs once after training on the selected checkpoint (`final_test=true`) and is never used for
  selection: this is the "general capability preserved" number to report.

### Ablations are CLI overrides

| change | override |
|---|---|
| drop one source | `data_module.mixture.sa1b=0` |
| drop a group | `data_module.drop_groups=[text_rich]` (all groups: `+experiment=sweeps/corpus_mixture`) |
| add application-domain data (egocentric frames, benchmark train images) | `data_module=general_vision_plus` |
| reweight | `data_module.mixture.cc12m=0.4 data_module.mixture.objects=0.05` |
| ImageNet-21k as the objects source | `objects_source=imagenet21k data_module.mixture.objects=0.20 data_module.mixture.cc12m=0.25` (after `download:imagenet21k corpus:imagenet21k`) |
| bigger val split | `corpus_split.val_fraction=0.005` |
| re-draw the splits (variance across splits) | `corpus_split.split_seed=1` |
| fewer images (fast sweeps) | `data_module=general_vision_lite` (~2.3M images, same splits) |

Adding a source takes three things: its directory, one entry in `sources.yaml`, and one YAML train
entry with its weight.

## Capture degradations (consistent teaching)

`collator.degradations` applies:

* motion blur;
* defocus blur;
* low light with sensor noise;
* rotation;
* finger / strap occlusion;
* JPEG artefacts.

Aggressive crops (`crop_scale`) and low budgets add the cut-off and "unmagnified subject" cases.
These are WearVQA's six image-quality failure modes.

* A degradation is sampled once per image, *before* the teacher/student branch, so the teacher labels
  exactly the degraded pixels the student sees.
* All randomness is stored in `TransformParams`, so cached features replay exactly.
* Degradations are off by default. Evaluation always uses clean images.

## Measuring "general capability preserved"

Benchmark accuracy alone does not show what was lost. Report:

* **Token fidelity on the held-out test split**, per group: `test/<group>/token_cos` and the LM-space cosine.
* **The grafted VLM on general VLM benchmarks**, alongside the seven project benchmarks.
* **The same numbers for the teacher**, at every token budget.

## Budget

| item | size |
|---|---|
| raw downloads: CC12M | ~1.2 TB |
| raw downloads: ImageNet-1k train (shared with the E2 probes) | 154 GB |
| raw downloads: ImageNet-21k (optional) | 1.19 TB |
| raw downloads: Open Images | ~0.55 TB |
| raw downloads: SA-1B (100 tars) | ~1 TB |
| raw downloads: text-rich | ~0.1 TB (estimate) |
| raw downloads: Ego4D 540ss, 1,000 h | ~1–1.5 TB (estimate) |
| raw downloads: EPIC-KITCHENS-100 videos | several hundred GB (estimate) |
| built corpus | ~2.5–3 TB |

Raw files can be deleted after step 2. Building is CPU-bound (JPEG decode, dHash, occasional
re-encode) and scales linearly with `WORKERS`. Time one source with `--limit 10000` before
scheduling the full build.
