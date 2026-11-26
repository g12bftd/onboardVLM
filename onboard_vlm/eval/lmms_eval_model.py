"""lmms-eval bridge for the general-capability benchmarks (E3): Gemma 4 with an optional grafted student.

Subclasses lmms-eval's own ``gemma4`` model (lmms-eval 0.7.4), so chat template, prompts, soft-token
budget, generation settings and scoring are exactly lmms-eval's for Gemma 4. The teacher and every
student run through this same class; the only difference is the vision tower. Results are therefore
comparable with published lmms-eval Gemma 4 numbers. Launch with ``run_lmms_eval.py``.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from lmms_eval.models.simple.gemma4 import Gemma4 as _LmmsGemma4
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError("the E3 general-benchmark bridge needs lmms-eval: pip install lmms-eval==0.7.4") from e

from onboard_vlm.utils.logging import get_logger

log = get_logger(__name__)


class OnboardGemma4(_LmmsGemma4):
    """``--model onboard_gemma4 --model_args pretrained=google/gemma-4-E2B-it,student_dir=<export>,max_soft_tokens=280``

    Args:
        pretrained: a supported Gemma 4 checkpoint id (or a local snapshot).
        revision: commit (default: the pinned one in ``SUPPORTED_TEACHERS``).
        student_dir: exported student; ``None`` = the original vision tower (teacher baseline).
        max_aspect: declared input domain of the student; out-of-domain inputs are counted and logged.
        blind: replace every image by a mid-grey image of the same size (same prompt, same number of soft tokens, no
            visual information): what the language model answers without seeing (onboard_vlm/eval/suites.py).
        **kwargs: everything lmms-eval's ``gemma4`` accepts (``max_soft_tokens``, ``batch_size``, ...).
    """

    def __init__(self, pretrained: str = "google/gemma-4-E2B-it", revision: str | None = None, student_dir: str | None = None,
                 max_aspect: float = 4.0, blind: bool | str = False, **kwargs: Any) -> None:
        self.blind = str(blind).lower() in ("1", "true", "yes")
        if not os.path.isdir(pretrained):  # pin the revision: lmms-eval's loader takes no revision argument
            from huggingface_hub import snapshot_download

            from onboard_vlm.teachers.gemma4 import resolve_revision

            rev = resolve_revision(pretrained, revision)
            log.info("===> lmms-eval: original Gemma 4 %s@%s", pretrained, rev[:7])
            pretrained = snapshot_download(pretrained, revision=rev)
        super().__init__(pretrained=pretrained, **kwargs)
        self.student_dir = student_dir
        self.grafted = None
        if student_dir:
            from onboard_vlm.geometry import InputDomain
            from onboard_vlm.students.io import load_student
            from onboard_vlm.vlm.grafted import GraftedVLM

            student, _ = load_student(student_dir)
            ref = next(self.model.parameters())
            self.grafted = GraftedVLM(self.model, student.to(ref.dtype), domain=InputDomain(max_aspect=float(max_aspect)))
            log.info("     grafted student %s into the unchanged VLM", student_dir)

    def _encode_image_data_url(self, image, *args: Any, **kwargs: Any):
        if self.blind:  # the vision-free control: identical structure, no visual content
            from PIL import Image

            image = Image.new("RGB", image.size, (128, 128, 128))
        return super()._encode_image_data_url(image, *args, **kwargs)

    def generate_until(self, requests):
        out = super().generate_until(requests)
        if self.grafted is not None:
            t = self.grafted.student_tower
            log.info("     student inputs: %d images, %d outside the declared input domain", t.images_seen, t.out_of_domain)
        return out
