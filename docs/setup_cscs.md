# Setup on CSCS Alps

The code lives in `$HOME` (`/users/$USER/repos/onboardVLM`). Data, outputs, caches and conda live on
scratch (`/capstor/scratch/cscs/$USER/onboardvlm`), with raw downloads strictly separated from
processed data (`scripts/paths.sh`):

```
/users/$USER/repos/onboardVLM/      git clone (code; $ONBOARD_VLM_CODE)
/capstor/scratch/cscs/$USER/onboardvlm/          ($ONBOARD_VLM_ROOT)
├── miniforge3/                     conda (envs: onboardvlm, onboardvlm-lmms)
├── data/
│   ├── raw/                        EXACTLY as downloaded, never modified
│   │   ├── vizwiz/ textvqa/ wearvqa/ egoblind/ egotextvqa/ egoschema/ cvbench/           benchmarks
│   │   ├── cc12m/ imagenet1k/ openimages/ sa1b/ pixmo_docs/ synthdog_en/ the_cauldron/   corpus images
│   │   ├── ego4d/ epic_kitchens/                                                         corpus videos
│   │   └── eval_sets/<set>/ nyuv2/                                                       evaluation / probes
│   └── processed/                  what the code reads
│       ├── benchmarks/<benchmark>/<split>.lmdb (+ .meta.json)
│       ├── corpus/<source>/shard_*.lmdb (+ .index.npz), keep.npz, dataset.json, eval_blocklist.json, CORPUS_CARD.md
│       └── eval_sets/<set>/shard_*.lmdb (+ .index.npz), nyuv2/
├── outputs/                        checkpoints, exported students, logs, evaluation results
├── cache/                          Hugging Face hub (teacher weights, HF token), pip, torch
└── logs/slurm/                     batch-job logs
```

`raw/imagenet1k` serves both the corpus (objects group) and the E2 probes. Scratch is cleaned
periodically (see the CSCS documentation for the current policy): copy `data/processed` and
`outputs` to project storage if you have it.

## 1. Clone

```bash
mkdir -p /users/$USER/repos && git clone git@github.com:g12bftd/onboardVLM.git /users/$USER/repos/onboardVLM
```

## 2. Conda environment (on a login node)

```bash
mkdir -p /capstor/scratch/cscs/$USER/onboardvlm && cd /capstor/scratch/cscs/$USER/onboardvlm
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
bash "Miniforge3-Linux-$(uname -m).sh" -b -p /capstor/scratch/cscs/$USER/onboardvlm/miniforge3
source /capstor/scratch/cscs/$USER/onboardvlm/miniforge3/etc/profile.d/conda.sh
conda env create -f /users/$USER/repos/onboardVLM/environment.yml
conda activate onboardvlm
cd /users/$USER/repos/onboardVLM && source scripts/paths.sh
```

PyPI's torch wheels are CUDA 13 builds, for x86_64 and aarch64 / Grace-Hopper alike. Check the GPU
driver first:

```bash
srun --account=<project> --partition=normal --time=00:05:00 nvidia-smi | grep "CUDA Version"
```

* 13.x or newer: `pip install torch torchvision`.
* 12.x: install a CUDA 12 build instead, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

Then:

```bash
pip install -e ".[data,dev]" && pip install ego4d awscli
srun --account=<project> --partition=normal --time=00:30:00 bash -c 'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))" && python -m pytest -q'
conda create -y --name onboardvlm-lmms --clone onboardvlm && conda run -n onboardvlm-lmms pip install "lmms-eval==0.7.4"
```

## 3. One-time access (on a login node, after `source scripts/paths.sh`)

| what | for | how |
|---|---|---|
| Hugging Face token | ImageNet-1k (gated, auto-approved), downloads | accept the terms on huggingface.co/datasets/timm/imagenet-1k-wds, then `hf auth login` (the token is stored in `$HF_HOME`, where the jobs find it) |
| Kaggle API token | EgoSchema videos | token from Kaggle → Settings → API, saved as `~/.kaggle/access_token` (chmod 600; a legacy `kaggle.json` also works); accept the `egoschema-public` competition rules |
| Ego4D licence | Ego4D frames | sign at ego4d-data.org (approval takes ~2 days); the email contains an AWS access key ID + secret key for the Ego4D S3 buckets (no AWS account needed). `aws configure --profile ego4d` (leave the region blank). The keys **expire 14 days after approval**: re-sign with the same email to reactivate them if `dl-ego4d` runs later |
| SA-1B licence | SA-1B | accept at ai.meta.com/datasets/segment-anything-downloads; save the link file as `$RAW_ROOT/sa1b/sa1b_links.txt` |
| OpenAI API key (optional) | not needed for any data step. Only the LLM-judged evaluation scores use it: EgoBlind, EgoTextVQA, WearVQA (`run_eval.py`) and MMBench, HallusionBench (lmms-eval) | platform.openai.com API account with billing (separate from ChatGPT); store the key in `~/.config/onboardvlm/secrets.sh` (chmod 600) and `source` it before submitting evaluation jobs. Without it, the judged scores are skipped and only the rule-based metrics are reported |

## 4. Data: every download and processing step as batch jobs

```bash
cd /users/$USER/repos/onboardVLM && git pull && conda activate onboardvlm && source scripts/paths.sh
bash scripts/slurm/submit_data_pipeline.sh --account <project> --dry-run      # prints the job graph
bash scripts/slurm/submit_data_pipeline.sh --account <project>
```

The script checks the credentials above, then submits one job per step
(`scripts/slurm/prepare_data.sbatch` → `make_datasets/prepare_all.sh <step>`). Each job gets
`afterok` dependencies on its inputs:

| jobs | runs after | raw → processed | size (approx.) |
|---|---|---|---|
| `models` | — | Gemma 4 E2B-it + E4B-it at the pinned revisions → `cache/huggingface` | 26 GB |
| `dl-<benchmark>` ×7 | — | VizWiz, TextVQA, WearVQA, EgoBlind, EgoTextVQA, EgoSchema, CV-Bench → `raw/<benchmark>` | a few hundred GB |
| `eval-sets` | — | 18 E1/E3 image sets → `raw/eval_sets/<set>` → `processed/eval_sets/<set>` | ~0.1 TB |
| `probe-data` | — | ImageNet-1k train → `raw/imagenet1k`; NYUv2 → `raw/nyuv2` → `processed/eval_sets/nyuv2` | 157 GB |
| `dl-<source>` ×5 | — | CC12M, Open Images, SA-1B (100 tars), PixMo-Docs + SynthDoG + Cauldron, EPIC-KITCHENS → `raw/<source>` | ~3.2 TB |
| `bench-extract`, `bench-lmdb` | benchmark downloads | `raw/<benchmark>` → `processed/benchmarks/<benchmark>/<split>.lmdb` | |
| `blocklist` | `bench-lmdb`, `eval-sets` | fingerprints of every evaluation image → `processed/corpus/eval_blocklist.json` | |
| `dl-ego4d` | `blocklist` | ~1,000 h of Ego4D 540p, benchmark source videos excluded → `raw/ego4d` | ~1–1.5 TB |
| `corpus-<source>` ×7 | its download + `blocklist` | `raw/<source>` → `processed/corpus/<source>` | ~2.5–3 TB in total |
| `corpus-finalize` | all `corpus-*` | global de-duplication, `dataset.json`, manifest, `CORPUS_CARD.md`, status report | |

Options:

* `--dl-partition <p>`: run the downloads on a data-transfer partition, if your cluster has one that
  reaches the internet (`sinfo -s`). Otherwise they use `--partition` (default `normal`), where each
  download job holds a whole node for its duration, charged to your allocation.
* `--time`: default 24 h.
* `--skip sa1b,ego4d`: leave out sources whose access is still pending. Everything else starts now;
  `corpus-finalize` waits until you re-run the script without `--skip` once the access is in place.

The submit script is safe to re-run at any time and only submits what is missing:

* a step that finished leaves `data/.steps/<step>.ok` and is not submitted again (delete the marker to
  redo the step);
* a job of the same name that is still queued or running is reused, never duplicated, and the new jobs
  depend on it;
* everything else is submitted.

If a job fails or times out, the jobs that depend on it are cancelled (`--kill-on-invalid-dep=yes`): fix
the cause and re-run the submit script. Downloads resume where they stopped, and an unfinished build is
redone from an empty directory. Monitor progress:

* `squeue -u $USER`;
* the job logs in `$SLURM_LOG_DIR`;
* `python scripts/check_data.py`, a summary of what is in place.

A single step can also be submitted on its own (this ignores the markers and runs the step):

```bash
sbatch --account=<project> --time=24:00:00 --output=$SLURM_LOG_DIR/%x-%j.out scripts/slurm/prepare_data.sbatch download:cc12m
```

After `corpus-finalize`, the raw corpus sources (`raw/{cc12m,openimages,sa1b,pixmo_docs,synthdog_en,the_cauldron,ego4d,epic_kitchens}`)
can be deleted. Keep `raw/imagenet1k`, `raw/eval_sets` and `raw/nyuv2`: the E2 probes read them.

ImageNet-21k is optional (`objects_source=imagenet21k`; [corpus.md](corpus.md#why-imagenet-1k-rather-than-imagenet-21k)):

```bash
OBJECTS_SOURCE=imagenet21k sbatch --account=<project> scripts/slurm/prepare_data.sbatch download:imagenet21k corpus:imagenet21k corpus_finalize
```
