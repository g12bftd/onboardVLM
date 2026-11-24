"""Prompt templates and Gemma 4 chat tokenisation.

Text inputs are produced by the *official* ``Gemma4Processor`` (chat template, ``boi + N x
image_token + eoi`` expansion, ``mm_token_type_ids`` for bidirectional vision attention), fed
with images that were already geometrically transformed by our collator, with
``do_resize=False``. Geometry therefore still happens exactly once per image.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import torch

LETTERS = "ABCDEFGHIJ"

# Templates. Fields: {question}, {options} (lettered list), {prompt} (dataset-provided prompt).
PROMPT_TEMPLATES: dict[str, str] = {
    # lmms-eval's VizWiz-VQA prompt (keeps 'unanswerable' answers in-distribution for the metric).
    "vizwiz_vqa": "{question}\nWhen the provided information is insufficient, respond with 'Unanswerable'.\n"
    "Answer the question using a single word or phrase.",
    "textvqa": "{question}\nAnswer the question using a single word or phrase.",
    "wearvqa": "{question}\nAnswer the question concisely in one sentence.",
    # Our blind-specific prompt (the EgoBlind paper reports blind-specific prompts help; theirs
    # is in the paper appendix and can be pasted here verbatim).
    "egoblind": "You are assisting a blind or visually impaired person who is wearing a camera. The frames come "
    "from their first-person video up to the moment they asked the question. Give a short, actionable answer "
    "that does not rely on them seeing anything.\nQuestion: {question}",
    # Verbatim from EgoTextVQA mllm_prompt.json (gpt-4o / gemini entries).
    "egotextvqa": "Based on the following images from a video, please briefly answer the following question with a "
    "single word, a phrase, or a short sentence. Question: {question}.  Output the answer to the question in the "
    "following format: Answer: [Full answer]. If you cannot answer the question, please answer 'Unanswerable' and "
    "briefly explain why you cannot answer.",
    "egoschema": "{question}\n{options}\nAnswer with the option's letter from the given choices directly.",
    # CV-Bench ships a ready-made prompt with the lettered choices.
    "cvbench": "{prompt}\nAnswer with the option's letter from the given choices directly.",
    "plain": "{question}",
}


def format_options(choices: Sequence[str] | None) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices or []))


def render_prompt(template: str, sample: dict[str, Any]) -> str:
    """``template`` is a key of :data:`PROMPT_TEMPLATES`, a literal format string, or ``"auto"``
    (pick the template named after ``sample["dataset"]``, falling back to ``"plain"``)."""
    if template == "auto":
        template = sample.get("dataset") if sample.get("dataset") in PROMPT_TEMPLATES else "plain"
    tpl = PROMPT_TEMPLATES.get(template, template)
    return tpl.format(
        question=sample.get("question", ""),
        options=format_options(sample.get("choices")),
        prompt=(sample.get("meta") or {}).get("prompt", sample.get("question", "")),
    )


def reference_answer(sample: dict[str, Any]) -> str:
    """Training target: MCQ letter, else the most frequent reference answer (ties -> first seen)."""
    if sample.get("choices") and sample.get("answer_index") is not None and sample["answer_index"] >= 0:
        return LETTERS[int(sample["answer_index"])]
    answers = [a for a in sample.get("answers") or [] if a]
    if not answers:
        return ""
    counts = Counter(answers)
    best = max(counts.values())
    return next(a for a in answers if counts[a] == best)


class Gemma4ChatInputs:
    """Callable that tokenises a batch of (visuals, sample) pairs with ``Gemma4Processor``.

    Args:
        processor: ``AutoProcessor.from_pretrained(<gemma-4 id>)`` (or any object with the same API).
        prompt: template name from :data:`PROMPT_TEMPLATES` or a literal format string.
        with_answer: append the reference answer as the assistant turn (teacher forcing for logit distillation)
            and return ``logits_to_keep`` / ``kd_mask`` / ``kd_targets`` for the answer span.
        system_prompt: optional system turn.
    """

    def __init__(self, processor: Any, prompt: str = "plain", with_answer: bool = False, system_prompt: str | None = None):
        self.processor = processor
        self.prompt = prompt
        self.with_answer = with_answer
        self.system_prompt = system_prompt
        processor.tokenizer.padding_side = "left"  # answers right-aligned -> one logits_to_keep tail

    def _messages(self, sample: dict[str, Any], kind: str, answer: str | None) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        msgs.append({"role": "user", "content": [{"type": kind}, {"type": "text", "text": render_prompt(self.prompt, sample)}]})
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def _call(self, texts: list[str], visuals: list[torch.Tensor], kind: str, budget: int, timestamps: list[list[float]] | None):
        kw: dict[str, Any] = dict(text=texts, return_tensors="pt", padding=True, do_resize=False, max_soft_tokens=budget)
        if kind == "image":
            kw["images"] = [[v] for v in visuals]
        else:
            from transformers.video_utils import VideoMetadata

            metas = []
            for v, ts in zip(visuals, timestamps or [None] * len(visuals)):
                ts = ts if ts is not None else [float(i) for i in range(v.shape[0])]
                # fps=1 with fractional "indices" == timestamps in seconds (Gemma prints mm:ss per frame).
                metas.append(VideoMetadata(total_num_frames=v.shape[0], fps=1.0, frames_indices=list(ts), height=v.shape[-2], width=v.shape[-1]))
            kw.update(videos=list(visuals), video_metadata=metas, do_sample_frames=False)
        return self.processor(**kw)

    def __call__(
        self,
        samples: Sequence[dict[str, Any]],
        visuals: Sequence[torch.Tensor],
        token_budget: int,
        timestamps: list[list[float]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """visuals: per sample ``UInt8[3, H, W]`` (image) or ``UInt8[T, 3, H, W]`` (video), already transformed."""
        kind = "video" if visuals[0].dim() == 4 else "image"
        tpl = self.processor.apply_chat_template
        prompts = [tpl(self._messages(s, kind, None), tokenize=False, add_generation_prompt=True) for s in samples]
        if not self.with_answer:
            return dict(self._call(prompts, list(visuals), kind, token_budget, timestamps))

        answers = [reference_answer(s) for s in samples]
        full = [tpl(self._messages(s, kind, a), tokenize=False, add_generation_prompt=False) for s, a in zip(samples, answers)]
        enc = dict(self._call(full, list(visuals), kind, token_budget, timestamps))
        prompt_lens = self._call(prompts, list(visuals), kind, token_budget, timestamps)["attention_mask"].sum(1)
        full_lens = enc["attention_mask"].sum(1)
        ans_lens = (full_lens - prompt_lens).clamp(min=1)
        # Left padding: answer tokens of row i occupy the last ans_lens[i] positions. Logits at
        # position p predict token p+1, so keep the last K = max(ans_len) + 1 positions.
        k = int(ans_lens.max()) + 1
        ids = enc["input_ids"]
        j = torch.arange(k)[None]
        enc["logits_to_keep"] = torch.tensor(k)
        enc["kd_mask"] = (j >= (k - 1 - ans_lens)[:, None]) & (j <= k - 2)
        targets = torch.full((ids.shape[0], k), -100, dtype=torch.long)
        targets[:, : k - 1] = ids[:, -(k - 1) :]
        enc["kd_targets"] = torch.where(enc["kd_mask"], targets, torch.full_like(targets, -100))
        return enc
