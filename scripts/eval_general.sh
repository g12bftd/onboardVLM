#!/usr/bin/env bash
# E3 general-capability suite (lmms-eval, identical settings for teacher and students).
#   bash scripts/eval_general.sh <run_name> [student_dir] [vlm_id] [budget]
# e.g. bash scripts/eval_general.sh teacher_e2b
#      bash scripts/eval_general.sh g4vit_l8_d384_e2b outputs/students/embedding_gemma4_vit_l8_d384_e2b
set -euo pipefail
RUN="${1:?run name}"; STUDENT="${2:-}"; VLM="${3:-google/gemma-4-E2B-it}"; BUDGET="${4:-280}"
OUT="${ONBOARD_VLM_OUT:-./outputs}/lmms_eval/${RUN}_b${BUDGET}"
ARGS="pretrained=${VLM},max_soft_tokens=${BUDGET}${STUDENT:+,student_dir=${STUDENT}}"
# perception & hallucination | general | OCR & documents | spatial | video | control (LM reasoning)
# IMAGE_TASKS / VIDEO_TASKS override the lists; DETERMINISTIC=1 drops the two tasks scored by a GPT judge in lmms-eval
# (mmbench_en_dev: answer extraction, hallusion_bench_image), so every score is rule-based and needs no OpenAI key.
IMAGE_TASKS="${IMAGE_TASKS:-pope,mmvp,hallusion_bench_image,mmbench_en_dev,mmstar,seedbench,realworldqa,ocrbench,docvqa_val,chartqa,infovqa_val,ai2d,blink,cv_bench,mmmu_val}"
VIDEO_TASKS="${VIDEO_TASKS:-videomme,mvbench}"
if [ -n "${DETERMINISTIC:-}" ]; then IMAGE_TASKS="$(echo ",$IMAGE_TASKS," | sed 's/,hallusion_bench_image,/,/; s/,mmbench_en_dev,/,/; s/^,//; s/,$//')"; fi
python run_lmms_eval.py --model onboard_gemma4 --model_args "$ARGS" --tasks "$IMAGE_TASKS" \
  --batch_size 1 --log_samples --output_path "$OUT/image"
# lmms-eval's gemma4 applies max_soft_tokens to video frames too: use Gemma 4's video budget (70 per frame)
VARGS="pretrained=${VLM},max_soft_tokens=70${STUDENT:+,student_dir=${STUDENT}}"
python run_lmms_eval.py --model onboard_gemma4 --model_args "$VARGS" --tasks "$VIDEO_TASKS" \
  --batch_size 1 --log_samples --output_path "$OUT/video"
