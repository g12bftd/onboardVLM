# Datasets: download -> LMDB

Two steps, as in TimeFM `cerebro_v2`: raw downloads into `$RAW_ROOT`, then
`make_datasets/lmdb/make_*.py` writes `$LMDB_ROOT/<dataset>/<split>.lmdb` plus
`<split>.meta.json` (keys, image sizes, frame counts, categories). The configs read
`data_root = $ONBOARD_VLM_DATA` (default `./data/lmdb`).

Tools needed: `wget`, `unzip`, `git`, and `pip install -U "huggingface_hub[cli]" gdown kaggle datasets av`.

```bash
export RAW_ROOT=/scratch/onboard_vlm/raw
```

```bash
export LMDB_ROOT=/scratch/onboard_vlm/lmdb
```

```bash
bash make_datasets/download/download_all.sh
```

```bash
bash make_datasets/build_all_lmdb.sh
```

Each `make_*.py` accepts `--dry_run` (parse and count only) and `--limit N` (debug subset).

## Per dataset

| Dataset | Raw source | Splits written | What is stored | Use in this project |
|---|---|---|---|---|
| VizWiz-VQA | vizwiz.cs.colorado.edu: `images/{train,val,test}.zip`, `vqa_data/Annotations.zip` | train 20,523 / val 4,319 / test 8,000 | image, question, 10 answers, `answer_type`, `answerable` | feature/embedding distillation images (train), logit-distillation QA (train), eval (val; test via EvalAI) |
| TextVQA 0.5.1 | dl.fbaipublicfiles.com/textvqa: `train_val_images.zip`, `test_images.zip`, `TextVQA_0.5.1_{train,val,test}.json` | train 34,602 / val 5,000 / test 5,734 questions | image, question, 10 answers, `question_id`, `image_classes` | same as VizWiz |
| WearVQA | HF `tonyliao-meta/WearVQA` (OpenReview link of the NeurIPS 2025 D&B paper) | test | image, question, answer(s), `quality_issues`, `domain`, `task_type` | eval only |
| EgoBlind | GitHub `doc-doc/EgoBlind` (CSVs) + Google Drive (videos) | test (`test.csv`, full: 2,558 answered Q), train (`finetune/train.csv`: 2,746 Q) | 16 frames in `[0, t_q]`, timestamps, 4 answers, `type` | distillation frames and QA (train), eval (test) |
| EgoTextVQA | GitHub `zhousheng97/EgoTextVQA` (CSVs) + HF `ShengZhou97/EgoTextVQA` or Drive (videos) | test; Outdoor high, Outdoor low, Indoor as separate LMDBs | 16 frames in `[0, t_q]`, answer, `question_type` | eval only (no train split) |
| EgoSchema | GitHub `egoschema/EgoSchema` (questions) + Kaggle `egoschema-public` (videos) | `subset` (500 with answers), `test` (5,031) | 32 frames per clip, 5 options, answer index (`-1` when hidden) | eval only |
| CV-Bench | HF `nyu-visionx/CV-Bench` | test 2,638 | image, question, choices, answer index, `task`, `source`, prompt | eval only |

**Contamination rule.** The default configs distil only on **train** splits (VizWiz, TextVQA,
EgoBlind). Label-free distillation on evaluation images would still be transductive, so keep
WearVQA, EgoTextVQA, EgoSchema, CV-Bench and all val/test splits out of training.

These train splits (about 45K images) are far too small and narrow to preserve the encoder's
general capabilities. [corpus.md](corpus.md) covers the general-vision corpus (OVLM-GV), its builders
and the benchmark decontamination blocklist.

## Source verification

Some hosts were reachable while this was written, so the following were checked directly:

* `EgoBlind/test.csv` (full test set; `test_half_release.csv` is the half released first) and `finetune/train.csv`: columns
  `question_id, video_name, question, answer0..3, type, start-time/s`. Questions without any reference answer are skipped.
* `EgoTextVQA/data/egotextvqa_{outdoor,indoor}_annotation.csv`: 4,848 rows / 694 videos and 2,216 rows / 813 videos.
* `EgoSchema/questions.json`: 5,031 items with `q_uid, question, option 0..4`; `subset_answers.json` maps `q_uid` to an int.

The VizWiz, TextVQA and Hugging Face URLs come from the official pages and dataset cards; those
hosts were not reachable from the build machine. If a URL has moved, the builders only need the
same files in the layout described in each `make_*.py` docstring.

* **WearVQA**: the release format could not be inspected, so the builder auto-detects
  columns. Run it with `--dry_run` first, and pass `--image_column/--question_column/...` if
  detection fails.
* **EgoBlind**: the name of the train CSV inside the Drive folder is not documented. Pass it
  explicitly with `--train_csv` (or `EGOBLIND_TRAIN_CSV=... bash make_datasets/build_all_lmdb.sh`).
  Videos are matched by `video_name` (`923`, `00923`, or `--video_pattern`).

## Record format

`onboard_vlm.data.lmdb_io.VQARecord` is pickled as a dict under key `sample_id`:

```
sample_id, dataset, split, media_type ("image"|"video"),
image: bytes | None, frames: list[bytes] | None, frame_timestamps: list[float] | None,
question, answers: list[str], choices: list[str] | None, answer_index: int | None,
answerable: int | None, answer_type: str | None, category: str | None, meta: dict
```

* Images are stored as their original JPEG/PNG bytes and are re-encoded only when the longest
  side exceeds `--max_side` (2048 by default). For a 16:9 frame at the largest budget (1120
  tokens), Gemma itself uses about 2112 px on the long side, so this loses almost nothing.
* EXIF orientation is applied on decode, as HF `load_image` does.
* Image-VQA keys are `"<image_id>::q<question_id>"`, so image mode reads each image once.
