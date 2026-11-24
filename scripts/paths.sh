# Source me:  source scripts/paths.sh
# Code lives in $HOME (not purged); data, outputs, caches and conda live on scratch, where raw downloads and
# processed (LMDB) data are kept strictly apart.
#
#   $ONBOARD_VLM_CODE = the clone this file is in (on Alps: $HOME/repos/onboardVLM)
#   $ONBOARD_VLM_ROOT = /capstor/scratch/cscs/$USER/onboardvlm
#     miniforge3/                 conda installation (envs: onboardvlm, onboardvlm-lmms)
#     data/raw/                   EXACTLY as downloaded: archives, tars, parquet, videos, repos (never modified)
#       vizwiz textvqa wearvqa egoblind egotextvqa egoschema cvbench          benchmarks
#       cc12m imagenet1k openimages sa1b pixmo_docs synthdog_en the_cauldron ego4d epic_kitchens    corpus
#       (imagenet1k also feeds the E2 probes; imagenet21k only with objects_source=imagenet21k)
#       eval_sets/<set>  nyuv2                                                evaluation / probe sources
#     data/processed/             what the code reads (one format per kind, built from data/raw)
#       benchmarks/<benchmark>/{train,val,test,...}.lmdb + .meta.json      (make_datasets/lmdb)
#       corpus/<source>/shard_*.lmdb + .index.npz, keep.npz, dataset.json   (make_datasets/corpus)
#       eval_sets/<set>/shard_*.lmdb + .index.npz, nyuv2/                   (make_datasets/eval_sets)
#     outputs/                    checkpoints, exported students, logs, evaluation results
#     cache/                      Hugging Face hub (model weights, HF token), pip, torch
#     logs/slurm/                 batch-job logs (scripts/slurm)
export ONBOARD_VLM_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # always the clone this file is in
export ONBOARD_VLM_ROOT="${ONBOARD_VLM_ROOT:-/capstor/scratch/cscs/${USER}/onboardvlm}"
export DATA_ROOT="$ONBOARD_VLM_ROOT/data"
export RAW_ROOT="$DATA_ROOT/raw"
export PROCESSED_ROOT="$DATA_ROOT/processed"
export LMDB_ROOT="$PROCESSED_ROOT/benchmarks"
export CORPUS_ROOT="$PROCESSED_ROOT/corpus"
export EVAL_ROOT="$PROCESSED_ROOT/eval_sets"
# the names the configs read (config/*.yaml: ${oc.env:...})
export ONBOARD_VLM_RAW="$RAW_ROOT"
export ONBOARD_VLM_DATA="$LMDB_ROOT"
export ONBOARD_VLM_CORPUS="$CORPUS_ROOT"
export ONBOARD_VLM_EVAL="$EVAL_ROOT"
export ONBOARD_VLM_OUT="$ONBOARD_VLM_ROOT/outputs"
# caches off $HOME (quota)
export HF_HOME="$ONBOARD_VLM_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$ONBOARD_VLM_ROOT/cache/pip"
export TORCH_HOME="$ONBOARD_VLM_ROOT/cache/torch"
export WORKERS="${WORKERS:-64}"          # parallel builder processes (a GH200 node has 4 x 72 Grace cores)
export SLURM_LOG_DIR="$ONBOARD_VLM_ROOT/logs/slurm"
mkdir -p "$RAW_ROOT" "$LMDB_ROOT" "$CORPUS_ROOT" "$EVAL_ROOT" "$ONBOARD_VLM_OUT" "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$SLURM_LOG_DIR"
echo "onboard-vlm: code=$ONBOARD_VLM_CODE | data root=$ONBOARD_VLM_ROOT | raw=$RAW_ROOT | processed=$PROCESSED_ROOT"
