# Setup on CSCS Alps (`/capstor/scratch/cscs/$USER`)

Everything lives under one root, `ONBOARD_VLM_ROOT=/capstor/scratch/cscs/$USER/onboardvlm`
(`scripts/paths.sh`). Raw downloads and processed data are kept strictly apart:

```
/capstor/scratch/cscs/$USER/onboardvlm/
├── code/                    git clone of this repository
├── miniforge3/              conda (envs: onboardvlm, onboardvlm-lmms)
├── data/
│   ├── raw/                 EXACTLY as downloaded, never modified
│   │   ├── vizwiz/ textvqa/ wearvqa/ egoblind/ egotextvqa/ egoschema/ cvbench/              benchmarks
│   │   ├── cc12m/ imagenet21k/ openimages/ sa1b/ pixmo_docs/ synthdog_en/ the_cauldron/     corpus (images)
│   │   ├── ego4d/ epic_kitchens/                                                            corpus (videos)
│   │   └── eval_sets/<set>/ imagenet1k/ nyuv2/                                              evaluation / probes
│   └── processed/           what the code reads
│       ├── benchmarks/<benchmark>/<split>.lmdb (+ .meta.json)                 make_datasets/lmdb
│       ├── corpus/<source>/shard_*.lmdb (+ .index.npz), keep.npz, dataset.json,
│       │   eval_blocklist.json, corpus_manifest.json, CORPUS_CARD.md          make_datasets/corpus
│       └── eval_sets/<set>/shard_*.lmdb (+ .index.npz), nyuv2/                make_datasets/eval_sets
├── outputs/                 checkpoints, exported students, logs, evaluation results
└── cache/                   Hugging Face hub (teacher weights), pip, torch
```

The scratch filesystem is cleaned periodically (see the CSCS documentation for the current policy).
Keep code pushed to GitHub. Copy `data/processed` and `outputs` to project storage if you have it.

## 1. Clone

```bash
mkdir -p /capstor/scratch/cscs/$USER/onboardvlm && cd /capstor/scratch/cscs/$USER/onboardvlm
git clone git@github.com:g12bftd/onboardVLM.git code
```

## 2. Conda environment

```bash
cd /capstor/scratch/cscs/$USER/onboardvlm
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
bash "Miniforge3-Linux-$(uname -m).sh" -b -p /capstor/scratch/cscs/$USER/onboardvlm/miniforge3
source /capstor/scratch/cscs/$USER/onboardvlm/miniforge3/etc/profile.d/conda.sh
conda env create -f code/environment.yml
conda activate onboardvlm
source code/scripts/paths.sh
```

PyTorch: PyPI's Linux wheels (x86_64 and aarch64 / Grace-Hopper) are CUDA 13 builds. Check the
driver on a GPU node first:

```bash
srun --account=<project> --partition=normal --time=00:05:00 nvidia-smi | grep "CUDA Version"
```

* `CUDA Version: 13.x` or newer: `pip install torch torchvision`.
* `12.x`: install a CUDA 12 build instead, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

Then:

```bash
cd /capstor/scratch/cscs/$USER/onboardvlm/code
pip install -e ".[data,dev]"
pip install ego4d awscli                 # Ego4D downloader (corpus)
srun --account=<project> --partition=normal --time=00:10:00 python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
srun --account=<project> --partition=normal --time=00:30:00 python -m pytest -q
```

The lmms-eval E3 bridge has many pinned dependencies, so it gets its own environment:

```bash
conda create -y --name onboardvlm-lmms --clone onboardvlm
conda activate onboardvlm-lmms && pip install "lmms-eval==0.7.4" && conda activate onboardvlm
```

## 3. One-time access

| what | needed for | how |
|---|---|---|
| Hugging Face token | gated sets, faster downloads | `hf auth login`, then accept the terms of `timm/imagenet-22k-wds` and `timm/imagenet-1k-wds` (and of `tonyliao-meta/WearVQA` if gated) on huggingface.co |
| Kaggle API key | EgoSchema videos | `~/.kaggle/kaggle.json` (chmod 600) and accept the rules of the `egoschema-public` competition |
| Ego4D licence | Ego4D frames (corpus) | sign at ego4d-data.org (approval takes days), then `aws configure --profile ego4d` with the keys you receive |
| SA-1B licence | SA-1B (corpus) | accept on ai.meta.com/datasets/segment-anything-downloads; save the link file as `$RAW_ROOT/sa1b/sa1b_links.txt` |
| OpenAI-compatible key | LLM judges of EgoBlind / EgoTextVQA / WearVQA (evaluation only) | `export OPENAI_API_KEY=...` |

## 4. Data: download → process

Every step is `bash make_datasets/prepare_all.sh <step>`, run from `code/` after
`source scripts/paths.sh`. The steps are idempotent: finished outputs are skipped, downloads resume,
and an unfinished build is redone from an empty directory. Downloads need outbound internet, so run
them on a login node inside `tmux`. Run the CPU-heavy processing as batch jobs:
`sbatch --account=<project> scripts/slurm/prepare_data.sbatch <step>`.

| # | step | where | raw → processed | size (approx.) |
|---|---|---|---|---|
| 1 | `models` | login | Gemma 4 E2B-it + E4B-it at the pinned revisions → `cache/huggingface` | 26 GB |
| 2 | `benchmarks_download` | login | 7 benchmarks → `raw/<benchmark>` | a few hundred GB (videos) |
| 3 | `benchmarks_extract` | login | unpack Google-Drive video archives (EgoBlind, EgoTextVQA) | |
| 4 | `benchmarks_lmdb` | sbatch | `raw/<benchmark>` → `processed/benchmarks/<benchmark>/<split>.lmdb` | |
| 5 | `eval_sets` | login (downloads) or sbatch | 18 E1/E3 image sets → `raw/eval_sets/<set>` → `processed/eval_sets/<set>` | ~0.1 TB |
| 6 | `probe_data` | login | ImageNet-1k train tars → `raw/imagenet1k`; NYUv2 → `raw/nyuv2` → `processed/eval_sets/nyuv2` | ~0.15 TB |
| 7 | `corpus_download` | login | CC12M, ImageNet-21k, Open Images, SA-1B, PixMo-Docs, SynthDoG, Cauldron, EPIC-KITCHENS → `raw/<source>` | ~5 TB |
| 8 | `blocklist` | sbatch | fingerprints of every evaluation image → `processed/corpus/eval_blocklist.json` | |
| 9 | `ego4d_download` | login | ~1,000 h of Ego4D 540p video, benchmark source videos excluded → `raw/ego4d` | ~1–1.5 TB |
| 10 | `corpus_build` (or `corpus:<source>`) | sbatch, one job per source | `raw/<source>` → `processed/corpus/<source>` | ~3.5–4 TB |
| 11 | `corpus_finalize` | sbatch | global de-duplication, `dataset.json`, manifest, `CORPUS_CARD.md` | |
| 12 | `check` | anywhere | prints what is in place (`scripts/check_data.py`) | |

The order matters:

* The blocklist (8) needs the benchmark LMDBs (4) and the evaluation sets (5).
* The Ego4D selection (9) and every corpus source (10) need the blocklist.
* If the blocklist is rebuilt (`FORCE_BLOCKLIST=1`), delete and rebuild the corpus sources.

After step 11, the raw corpus sources (`raw/{cc12m,imagenet21k,openimages,sa1b,pixmo_docs,synthdog_en,the_cauldron,ego4d,epic_kitchens}`)
can be deleted. Keep `raw/eval_sets`, `raw/imagenet1k` and `raw/nyuv2`: the E2 probes read their
labels from there.
