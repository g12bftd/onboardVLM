#!/usr/bin/env bash
# E3: general vision-language capabilities with lmms-eval 0.7.4 (onboard_vlm/eval/suites.py): the Cambrian-1 suite
# (General, Knowledge, OCR & Chart, Vision-Centric) + POPE, MMStar, InfoVQA, identical settings for teacher and students.
#
#   bash scripts/eval_general.sh <run_name> [student_dir] [vlm_id] [budget]
#   bash scripts/eval_general.sh teacher_e2b                                   # the original encoder
#   bash scripts/eval_general.sh shrink_L8_D768_s0 $ONBOARD_VLM_OUT/students/shrink_e2b_it_L8_D768_teacher_uniform-uniform_s0
#   BLIND=1 bash scripts/eval_general.sh blind_e2b                             # every image grey: the language-only floor
#
# Rule-based scores only by default; JUDGED=1 adds MMBench, MathVista, HallusionBench (GPT judge, needs OPENAI_API_KEY).
# TASKS=<comma list> overrides the suite; VIDEO=1 also runs Video-MME and MVBench (70 tokens per frame, Gemma 4's video
# budget). Outputs: $ONBOARD_VLM_OUT/lmms_eval/<run_name>_b<budget>/ (results json + per-sample logs for paired tests).
set -euo pipefail
RUN="${1:?run name}"; STUDENT="${2:-}"; VLM="${3:-google/gemma-4-E2B-it}"; BUDGET="${4:-280}"
OUT="${ONBOARD_VLM_OUT:-./outputs}/lmms_eval/${RUN}_b${BUDGET}"
EXTRA_ARGS="${BLIND:+,blind=True}"
ARGS="pretrained=${VLM},max_soft_tokens=${BUDGET}${STUDENT:+,student_dir=${STUDENT}}${EXTRA_ARGS}"
TASKS="${TASKS:-$(python -m onboard_vlm.eval.suites ${JUDGED:+--judged})}"   # the suite list needs only onboard_vlm
echo "[eval_general] $RUN | budget $BUDGET | ${STUDENT:-original encoder}${BLIND:+ | BLIND} | tasks: $TASKS"
PY="${ONBOARD_VLM_LMMS_PYTHON:-}"   # lmms-eval lives in its own env (docs/setup_cscs.md)
if [ -z "$PY" ]; then  # the env by name, wherever it lives (conda's envs_dirs)
  PY="$("${CONDA_EXE:-conda}" run -n "${ONBOARD_VLM_LMMS_ENV:-onboardvlm-lmms}" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1 || true)"
  [ -x "$PY" ] || PY=python
fi
"$PY" run_lmms_eval.py --model onboard_gemma4 --model_args "$ARGS" --tasks "$TASKS" \
  --batch_size 1 --log_samples --output_path "$OUT/image"
if [ -n "${VIDEO:-}" ]; then
  VARGS="pretrained=${VLM},max_soft_tokens=70${STUDENT:+,student_dir=${STUDENT}}${EXTRA_ARGS}"
  "$PY" run_lmms_eval.py --model onboard_gemma4 --model_args "$VARGS" --tasks "videomme,mvbench" \
    --batch_size 1 --log_samples --output_path "$OUT/video"
fi
