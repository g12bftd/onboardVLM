# OVLM-GV: the general-vision distillation corpus

Feature distillation is label-free: the student learns the teacher's function only on the images
it is shown. What the student keeps of the Gemma 4 encoder's general ability is therefore decided by
the image distribution of the corpus. OVLM-GV is built to cover that distribution along seven axes,
using only public, versioned sources, with every preprocessing step scripted and logged.

| group | source (directory under `$CORPUS_ROOT`) | covers | images (raw) | licence / access | weight | precedent |
|---|---|---|---:|---|---:|---|
| web | CC12M, pixparse snapshot (`cc12m`) | open-vocabulary web photos, illustrations, products | 10,968,539 | CC12M terms; HF, open | 0.25 | MobileCLIP (DataCompDR), CLIP-KD |
| objects | ImageNet-21k fall11 (`imagenet21k`) | 21k fine-grained object categories | 13,673,551 | ImageNet terms (non-commercial research); HF, gated | 0.20 | TinyViT (IN-21k distillation) |
| scenes | Open Images V7, box-annotated train subset (`openimages`) | cluttered multi-object scenes, spatial layout | 1,743,042 | images CC BY 2.0, annotations CC BY 4.0; S3, open | 0.10 | |
| scenes | SA-1B, first 100 of 1,000 tars (`sa1b`) | high-resolution dense scenes | ~1.1M | SA-1B Research License; form | 0.08 | EfficientSAM / MobileSAM |
| text-rich | PixMo-Docs (`text_rich/pixmo_docs`) | charts, tables, diagrams, documents | 255,261 | ODC-BY 1.0; HF, open | 0.06 | Molmo |
| text-rich | SynthDoG-en (`text_rich/synthdog_en`) | dense rendered text | 500,000 | MIT; HF, open | 0.03 | Donut |
| text-rich | The Cauldron, 13 document / chart / scene-text / UI subsets (`text_rich/cauldron/*`) | real documents, infographics, scene text, screenshots | ~0.4M (4 synthetic subsets capped at 50k) | per subset; HF, open | 0.08 | Idefics2 |
| egocentric | Ego4D v2 `video_540ss`, ~1,000 h scenario-stratified, 0.5 fps (`egocentric/ego4d`) | head-mounted viewpoint, hands, motion blur, indoor/outdoor daily life | ≤1.8M frames | Ego4D licence; signed agreement + AWS keys | 0.14 | |
| egocentric | EPIC-KITCHENS-100, 0.5 fps (`egocentric/epic_kitchens`) | egocentric manipulation, kitchens | ~0.18M frames | CC BY-NC 4.0; open | 0.03 | |
| target | VizWiz, TextVQA, EgoBlind **train** splits (`target_train`) | blind-user photos, scene text, smart-glasses video | ~45K | per benchmark | 0.03 | |

The weights are the sampling probabilities in `config/data_module/general_vision.yaml`
(`data_module.mixture`). After filtering, decontamination and de-duplication, the exact counts are
written to `$CORPUS_ROOT/CORPUS_CARD.md` and `corpus_manifest.json`. Report those numbers in the
paper, not the raw ones above.

## Why this corpus is defensible

* **Coverage, not just scale.** Each group exists because a capability of the teacher, and of the
  target benchmarks, needs it:
  * web and object-centric images: recognition (VizWiz "what is this");
  * multi-object and high-resolution scenes: counting and spatial relations (CV-Bench), plus detail
    at large token budgets;
  * text-rich images: OCR (TextVQA, EgoTextVQA);
  * egocentric frames: the smart-glasses viewpoint (WearVQA, EgoBlind, EgoSchema).

  Each group can be removed in a leave-one-group-out ablation
  (`+experiment=sweeps/corpus_mixture`), so its contribution is measured, not assumed.
* **Every source has precedent in encoder distillation.**
  * ImageNet-21k: TinyViT.
  * CC12M / DataComp-style web data: MobileCLIP and RADIO.
  * SA-1B: EfficientSAM and MobileSAM.
  * Ego4D: the standard egocentric pre-training source.
* **Reproducible.** All sources are fixed snapshots: wds tars, parquet files, S3 tars and versioned
  video sets. No URL lists are re-scraped. LAION is excluded: it was withdrawn after CSAM reports, and
  Re-LAION would be one more URL-scraped set. DataComp-12M is excluded because it ships URLs only, so
  link rot would make the corpus differ between labs. The finalized training set is identified by a
  SHA-256 digest over the sorted SHA-1 of every training image (`train_sha1_digest` in the manifest).
* **Decontaminated against every evaluation set**, by content and by identity (see below), before
  de-duplication.
* **Licences are tracked per source.** The corpus is for research. ImageNet-21k, SA-1B, Ego4D and
  EPIC-KITCHENS are non-commercial / research-only, so a model distilled on the full mixture inherits
  those terms. The leave-one-group-out configs let you report a commercially cleaner variant if needed.

## Pipeline

```
download (make_datasets/download/corpus/*.sh) -> $RAW_ROOT
benchmark LMDBs (make_datasets/build_all_lmdb.sh) -> $LMDB_ROOT
make_datasets/build_corpus.sh
   1 blocklist  : fingerprints of every benchmark val/test image + frame, TextVQA ids, EgoSchema videos
   2 sources    : decode -> filter -> decontaminate -> (downscale) -> LMDB shards + fingerprint index
   3 finalize   : global exact + perceptual de-dup, per-source held-out split, manifest + corpus card
train            : data_module=general_vision (or general_vision_lite)
```

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
hf auth login && bash make_datasets/download/corpus/download_imagenet21k.sh   # accept the terms on the HF page first (~1.2 TB)
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
| ImageNet-21k | accept the terms on huggingface.co/datasets/timm/imagenet-22k-wds |
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
WORKERS=32 bash make_datasets/build_corpus.sh text_rich openimages sa1b imagenet21k cc12m ego4d epic_kitchens
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
  keep.npz / holdout.npz          train / held-out masks per shard (finalize)
```

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
3. **Held-out split**: 2,000 images per source directory (seeded) are removed from training. They serve
   as in-distribution validation (`ImageCorpusDataset(split="holdout")`), so you can report how well
   the student matches the teacher on general images it never saw, per group.

Outputs:

* `keep.npz` / `holdout.npz` per source;
* `corpus_manifest.json`: counts per source; within- vs cross-source duplicates; build filter
  statistics; licences; the train-set digest;
* `CORPUS_CARD.md`: the same as a table for the paper's appendix.

The paper configs set `require_finalized: true`, so a corpus that has not been finalized cannot be
used by accident.

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

* **Sampling** is a fixed mixture (`MixtureSampler`). Each epoch draws `samples_per_epoch` (2M)
  images: multinomially over components by weight, then uniformly without replacement within a
  component. It is deterministic given seed + epoch, identical on every DDP rank, and partitioned by
  Lightning's distributed wrapper. Unlike `WeightedRandomSampler`, it has no 2^24-category limit.
* **Repetition**. At start-up the data module logs, per component, the draws per epoch and how many
  passes over its images that means. With the default 30 × 2M draws:
  * web, objects and scenes: 1–4 passes;
  * text-rich and egocentric: 4–14 passes;
  * target train splits: ~40 passes.

  Every pass is a new random crop at a sampled token budget. Following "a good teacher is patient
  and consistent" (Beyer et al., 2022), teacher and student always see the same augmented view.
* **Ablations**:
  * `data_module.mixture.<component>=0` drops one component;
  * `data_module.drop_groups=[egocentric]` drops a group;
  * `data_module=general_vision_lite` uses ~2.3M images (the first shards of every component), with
    identical filtering, de-duplication and held-out split, for sweeps.
* **Validation**:
  * held-out images of every group (web + objects + scenes, text-rich, egocentric);
  * 1,000 VizWiz val images.

  Both are decontaminated from training by construction.

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

* **Token fidelity on held-out general images**, per group: `val/token_cos` and the LM-space cosine.
* **The grafted VLM on general VLM benchmarks**, alongside the seven project benchmarks.
* **The same numbers for the teacher**, at every token budget.

## Budget

| item | size |
|---|---|
| raw downloads: CC12M | ~1.2 TB |
| raw downloads: ImageNet-21k | ~1.2 TB |
| raw downloads: Open Images | ~0.55 TB |
| raw downloads: SA-1B (100 tars) | ~1 TB |
| raw downloads: text-rich | ~0.1 TB (estimate) |
| raw downloads: Ego4D 540ss, 1,000 h | ~1–1.5 TB (estimate) |
| raw downloads: EPIC-KITCHENS-100 videos | several hundred GB (estimate) |
| built corpus | ~3.5–4 TB |

Raw files can be deleted after step 2. Building is CPU-bound (JPEG decode, dHash, occasional
re-encode) and scales linearly with `WORKERS`. Time one source with `--limit 10000` before
scheduling the full build.
