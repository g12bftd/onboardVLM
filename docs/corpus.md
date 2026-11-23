# Distillation corpus

**Short answer: the benchmark train splits are not enough.** VizWiz train, TextVQA train and
EgoBlind train frames together hold roughly 45K distinct images from three narrow domains. The
Gemma 4 vision encoder (167M parameters, shared by E2B and E4B) was trained for general
image-text understanding. To keep that generality, a much smaller student needs to see the
teacher's behaviour on a broad, large, decontaminated distribution:

* **Breadth**: millions of unique images across natural scenes, objects, text and documents,
  screens and egocentric video.
* **Resolution**: native high-resolution images for the 560 and 1,120-token budgets.
* **Domain**: smart-glasses capture, including the degraded images glasses actually produce.

The pixels need no labels: the teacher provides all targets.

## Recommended mixture

`config/data_module/stage1_general.yaml` implements this. Weights are sampling probabilities;
tune them per experiment.

| Group | Weight | Sources (license / access) | Why |
|---|---|---|---|
| General web images | 0.40 | DataComp-12M / DataCompDR-12M (`mlfoundations/DataComp-12M`, `apple/DataCompDR-12M`: URL metadata, fetch with img2dataset), CC12M (`pixparse/cc12m-wds`, images included), optionally COYO-700M or DataComp-1B subsets | Broad concept coverage. DataCompDR is what MobileCLIP used to distil CLIP ensembles into mobile encoders. |
| High resolution | 0.10 | SA-1B (11M images about 1500x2250, Meta research license), OpenImages V7 (CC BY 2.0), Objects365 | The 560 / 1,120-token budgets need >= 1.3 / 2.6 MP inputs. Web images are mostly < 0.5 MP and get *upsampled* by Gemma's resize. |
| Text-rich | 0.18 | The Cauldron (`HuggingFaceM4/the_cauldron`: DocVQA, ChartQA, InfographicVQA, AI2D, ST-VQA, TextCaps, screen2words, rendered_text ... train images), PixMo-Docs (`allenai/pixmo-docs`), SynthDoG-en (`naver-clova-ix/synthdog-en`), TextOCR | OCR is the capability students lose first, and TextVQA, EgoTextVQA and WearVQA depend on it. |
| Egocentric / glasses | 0.22 | Ego4D (Ego4D license, `ego4d` CLI), EPIC-KITCHENS-100 (CC BY-NC), Ego-Exo4D ego views, HoloAssist, Project Aria datasets (Aria Everyday Activities, Nymeria: real smart-glasses capture), EgoObjects | Target domain: head-mounted viewpoint, motion, hands, clutter. |
| Accessibility / benchmark train | 0.10 | VizWiz-VQA train, TextVQA train, EgoBlind train (already built by `make_datasets/lmdb`) | In-domain blind-user photos and scene text, with no labels used. |

**Scale.**

* Unique images: aim for at least 10M.
* Samples seen: aim for 0.5–2B (for example 2M samples/epoch for 300 epochs). Beyer et al.
  (2022) show that distillation keeps improving with very long, consistent-view schedules.
* Teacher cost: one teacher forward at 280 tokens is about 0.54 TMAC (`run_profile.py`). 600M
  samples is therefore on the order of 10^3 H100-hours for the teacher alone.
* Cache vs online: an offline cache costs about 430 KB per view at 280 tokens (bf16), so
  10M views is about 4.3 TB. Use `run_cache_teacher.py` for ablation subsets (1–2M images)
  and the online teacher for full runs.

## Capture degradations (consistent teaching)

`collator.degradations` applies motion blur, defocus blur, low light with sensor noise,
rotation, finger/strap occlusion and JPEG artefacts. Aggressive crops (`crop_scale`) and low
budgets add the cut-off and "unmagnified subject" cases. These are WearVQA's six image-quality
failure modes.

* The degradation is sampled once per image, *before* the teacher/student branch, so the
  teacher labels exactly the degraded pixels the student sees (Beyer et al.'s "consistent
  teaching").
* All randomness is stored in `TransformParams`, so cached features replay exactly.
* Evaluation always uses clean images.

## Decontamination

The corpus must not contain benchmark validation or test images: label-free distillation on
evaluation images is still transductive leakage. Run this before building any corpus:

```bash
python make_datasets/corpus/make_eval_blocklist.py --lmdb_dirs $LMDB_ROOT/vizwiz $LMDB_ROOT/textvqa $LMDB_ROOT/wearvqa $LMDB_ROOT/egoblind $LMDB_ROOT/egotextvqa_outdoor_high $LMDB_ROOT/egotextvqa_indoor $LMDB_ROOT/egoschema $LMDB_ROOT/cvbench --egoschema_uid_map $RAW_ROOT/egoschema/repo/uid_to_ego4d.json --output $CORPUS_ROOT/eval_blocklist.json
```

This records SHA-1 and 64-bit dHash fingerprints of every non-train benchmark image and stored
frame. It also lists the 5,031 Ego4D source clips behind EgoSchema; EgoTextVQA-Indoor reuses
those clips. Every corpus builder then drops:

* exact duplicates and near-duplicates (dHash within 4 bits) of those images;
* whole Ego4D source videos (`--exclude_mode video`, default) or just the clips' seconds (`span`).

Known overlaps the blocklist catches:

* TextVQA images come from OpenImages, so an OpenImages corpus contains TextVQA val/test images.
* CV-Bench 2D uses COCO and ADE20K images.
* EgoSchema and EgoTextVQA-Indoor are Ego4D clips.

## Building corpora

Each builder writes `shard_*.lmdb` and `shard_*.meta.json`, which the `image_corpus` dataset
reads. Builders filter by size and aspect ratio, check the blocklist, optionally de-duplicate
within the corpus (`--dedup`), and can run as N parallel workers
(`--worker_id i --num_workers N`).

```bash
python make_datasets/corpus/make_image_corpus.py --source webdataset --input_path "/data/cc12m-wds/*.tar" --name cc12m --output_path $CORPUS_ROOT/general/cc12m --blocklist $CORPUS_ROOT/eval_blocklist.json --dedup
```

```bash
python make_datasets/corpus/make_image_corpus.py --source hf --input_path HuggingFaceM4/the_cauldron --hf_config chartqa --image_column images --name cauldron_chartqa --output_path $CORPUS_ROOT/text_rich/docs_charts --blocklist $CORPUS_ROOT/eval_blocklist.json
```

```bash
python make_datasets/corpus/make_image_corpus.py --source folder --input_path /data/sa1b/images --name sa1b --output_path $CORPUS_ROOT/highres/sa1b --min_side 1024 --blocklist $CORPUS_ROOT/eval_blocklist.json
```

```bash
python make_datasets/corpus/make_frame_corpus.py --input_path /data/ego4d/v2/full_scale --fps 0.5 --name ego4d --output_path $CORPUS_ROOT/egocentric/ego4d --blocklist $CORPUS_ROOT/eval_blocklist.json --worker_id 0 --num_workers 8
```

The frame builder samples every `1/fps` seconds with seeks, which is cheap on hour-long videos.
It skips near-static frames (dHash within `--still_bits` of the previous kept frame) and drops
benchmark source videos.

## Measuring "general capability preserved"

Benchmark accuracy alone does not show what was lost. Report three kinds of evidence:

* **Token fidelity on a held-out general set**: `val/token_cos` and LM-space cosine on images
  never used in training.
* **The grafted VLM on general VLM benchmarks**, alongside the seven project benchmarks.
* **The same numbers for the teacher**, at every token budget.
