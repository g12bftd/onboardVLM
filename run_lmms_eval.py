"""E3 general-capability benchmarks through lmms-eval (pip install lmms-eval==0.7.4).

Registers ``onboard_gemma4`` (onboard_vlm.eval.lmms_eval_model.OnboardGemma4) and runs lmms-eval's own
CLI, so every lmms-eval flag works. Teacher and students use the same class:

    python run_lmms_eval.py --model onboard_gemma4 --model_args pretrained=google/gemma-4-E2B-it,max_soft_tokens=280 \\
        --tasks pope,mmbench_en_dev --batch_size 1 --log_samples --output_path outputs/lmms_eval/teacher_e2b
    python run_lmms_eval.py --model onboard_gemma4 \\
        --model_args pretrained=google/gemma-4-E2B-it,student_dir=outputs/students/embedding_gemma4_vit_l8_d384_e2b,max_soft_tokens=280 \\
        --tasks pope,mmbench_en_dev --batch_size 1 --log_samples --output_path outputs/lmms_eval/g4vit_l8_d384_e2b

``scripts/eval_general.sh`` runs the full E3 suite for one model. Keep ``--log_samples``: per-sample
outputs are what ``scripts/aggregate_results.py`` uses for bootstrap CIs and paired tests.
"""

from __future__ import annotations


def main() -> None:
    from lmms_eval.__main__ import cli_evaluate
    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest

    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(model_id="onboard_gemma4", simple_class_path="onboard_vlm.eval.lmms_eval_model.OnboardGemma4"), overwrite=True
    )
    cli_evaluate()


if __name__ == "__main__":
    main()
