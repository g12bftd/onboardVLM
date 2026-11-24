"""Teacher-generated targets for logit distillation (sequence-level KD).

Logit distillation compares next-token distributions of the VLM with the teacher's tower and with
the student, teacher-forced on a response. If that response is a human reference answer, the KL is
measured on sequences the teacher would not produce itself. The protocol therefore distils on the
teacher's *own* responses: ``run_generate_targets.py`` prompts the original Gemma 4 VLM (greedy,
bit-exact processor geometry) and caches prompt + response per image in LMDB; this module provides
the prompting dataset and the reader.
"""

from __future__ import annotations

import glob
import hashlib
import os
from typing import Any, Sequence

from torch.utils.data import ConcatDataset, Dataset

from onboard_vlm.data.datasets import LMDBVQADataset
from onboard_vlm.data.prompts import render_prompt
from onboard_vlm.registry import build_dataset, register_dataset

# Image-agnostic prompts covering the capabilities the vision encoder must preserve.
DEFAULT_TEMPLATES: dict[str, str] = {
    "caption_detailed": "Describe this image in detail.",
    "caption_short": "Describe this image in one sentence.",
    "ocr": "Read out all the text visible in this image. If there is no text, say that there is none.",
    "spatial": "List the main objects in this image and describe where they are relative to each other.",
    "assistive": "I am blind. Briefly describe what is in front of me and anything I should be careful about.",
    "salient": "What is the main subject of this image, and what is happening?",
}


def _stable_index(key: str, n: int, seed: int) -> int:
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n


class PromptedImages(Dataset):
    """Pairs every image of ``base`` with prompts.

    Args:
        base: dataset config (images; e.g. a corpus source or a benchmark train split).
        templates: ``{name: text}``; ``None`` = :data:`DEFAULT_TEMPLATES`.
        templates_per_image: how many templates each image gets (chosen by a seeded hash of its id).
        use_dataset_question: use the dataset's own question (rendered with its benchmark prompt
            template) instead of the templates (benchmark train splits).
    """

    def __init__(
        self,
        base: dict[str, Any],
        templates: dict[str, str] | None = None,
        templates_per_image: int = 1,
        use_dataset_question: bool = False,
        seed: int = 0,
    ):
        self.base = build_dataset(base)
        self.templates = dict(templates or DEFAULT_TEMPLATES)
        self.names = sorted(self.templates)
        self.k = 1 if use_dataset_question else max(1, min(templates_per_image, len(self.names)))
        self.use_dataset_question, self.seed = use_dataset_question, seed

    def __len__(self) -> int:
        return len(self.base) * self.k

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = dict(self.base[i // self.k])
        if self.use_dataset_question:
            s["question"], s["template"] = render_prompt("auto", s), f"dataset:{s.get('dataset')}"
        else:
            first = _stable_index(str(s["sample_id"]), len(self.names), self.seed)
            name = self.names[(first + i % self.k) % len(self.names)]
            s["question"], s["template"] = self.templates[name], name
        s["source_id"] = s["sample_id"]
        s["sample_id"] = f"{s['template']}/{s['sample_id']}"
        return s


@register_dataset("teacher_responses")
class TeacherResponseDataset(ConcatDataset):
    """Reads every ``part_*.lmdb`` written by ``run_generate_targets.py`` under ``path`` (question =
    the exact prompt, ``answers[0]`` = the teacher's response); use with ``prompt: plain``."""

    def __init__(self, path: str, limit: int | None = None, templates: Sequence[str] | None = None):
        parts = sorted(os.path.basename(p)[: -len(".lmdb")] for p in glob.glob(os.path.join(path, "part_*.lmdb")))
        if not parts:
            raise FileNotFoundError(f"no part_*.lmdb under {path}: run run_generate_targets.py first")
        cats = list(templates) if templates is not None else None
        super().__init__([LMDBVQADataset(path, split=p, mode="vqa", categories=cats) for p in parts])
        self.sizes = [s for d in self.datasets for s in d.sizes]
        if limit is not None:
            self.datasets, self.sizes = _truncate(self.datasets, limit), self.sizes[:limit]
            self.cumulative_sizes = self.cumsum(self.datasets)


def _truncate(datasets: list, limit: int) -> list:
    out, left = [], limit
    for d in datasets:
        if left <= 0:
            break
        if len(d) > left:
            d.index, d.sizes = d.index[:left], d.sizes[:left]
        out.append(d)
        left -= len(d)
    return out
