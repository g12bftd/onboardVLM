# Benchmarks and official metrics

All benchmarks run through the same path: LMDB -> `VLMEvalCollator` (deterministic Gemma
geometry once per image/frame) -> official `Gemma4Processor` chat template -> `generate()` on
either the stock VLM (teacher baseline) or a `GraftedVLM` (student). `run_eval.py` writes
`predictions.jsonl`, `metrics.json` and, where the benchmark uses a server, the official
submission file.

| Benchmark | Split scored locally | Official metric | Our implementation | Submission file |
|---|---|---|---|---|
| **VizWiz-VQA** | `val` (4,319) | VQA accuracy over 10 crowd answers; Answerability: average precision | `vqa_soft_accuracy` (EvalAI normalisation + `vqaEval.py` leave-one-out), per `answer_type`; AP on `1 - p("Unanswerable" \| prompt)` | `vizwiz_vqa_submission.json`, `vizwiz_answerability_submission.json` (test -> EvalAI) |
| **TextVQA** | `val` (5,000) | VQA accuracy over 10 answers | same as VizWiz | `textvqa_submission.json` (test -> EvalAI) |
| **WearVQA** | `test` (~2.5K) | LLM-as-judge QA accuracy | binary judge accuracy overall and per quality issue / domain / task type; proxies: normalised exact match, containment | - |
| **EgoBlind** | `test.csv`, the full test set (2,558 answered Q / 656 videos; the first public half `test_half_release` has 1,283 Q) | GPT-4o-mini judge vs 4 references: accuracy (% "yes") and mean score (0-5), per question type | verbatim prompt + `calculate_metrics` replica | `egoblind_pred.jsonl` (input format of the official `eval.py`) |
| **EgoTextVQA** | Outdoor 4,848 (high & low res), Indoor 2,216 | GPT-4o-mini judge: accuracy = yes/(yes+no) and mean integer score (0-5), per question type | verbatim prompt, lower-casing/newline removal and `calculate_score` replica; ANLS & EM as auxiliary | `egotextvqa_<subset>_pred.json` (input format of `gpt_eval.py`) |
| **EgoSchema** | `subset` (500 public answers) | 5-way MCQ accuracy | letter / index / option-text parsing | `egoschema_kaggle.csv` (`q_uid,answer`) + `egoschema_submission.json` for the full 5,031 |
| **CV-Bench** | `test` (2,638) | MCQ accuracy; overall = `((acc_ADE20K + acc_COCO)/2 + acc_Omni3D)/2` | per source and task + official overall | - |

## Details

### VQA accuracy (VizWiz-VQA, TextVQA)

For a prediction `p` and the 10 human answers `a_1..a_10`:

    acc(p) = (1/10) * sum_i min(1, |{j != i : a_j == p}| / 3)

The prediction is first normalised by EvalAI's answer processor (lower-case, punctuation,
number words to digits, articles, contractions). When the 10 answers are not unanimous, the
answers and the prediction are additionally passed through `processPunctuation` and
`processDigitArticle`, as `vqaEval.py` does. Reference-code quirks are kept so local numbers
match the servers. Test-split answers are hidden for both datasets, so local numbers come from
`val`, and test numbers come from uploading the submission file to EvalAI.

**VizWiz answerability** is the average precision of an answerability score against the
`answerable` labels. We use `1 - p_model("Unanswerable" | prompt)`, computed with one
teacher-forced forward pass (`eval.runner.continuation_logprob`). Our AP matches
`sklearn.metrics.average_precision_score` (tested).

### LLM-judged benchmarks

The judge is any OpenAI-compatible chat-completions endpoint (default `gpt-4o-mini` at
api.openai.com, key read from `OPENAI_API_KEY`). Judgements are cached in
`judge_cache.jsonl`, so a run can be resumed. Point `judge.base_url` at a local vLLM server to
use an open judge; this changes the protocol, so report it as such.

* **EgoBlind**: the system and user prompts are copied verbatim from `doc-doc/EgoBlind/eval.py`.
  All four references are shown, and missing ones are rendered as `nan` exactly as the official
  pandas-based script does. 1,243 of the 1,283 released questions have fewer than four
  references. The six question types are information reading, safety warnings, navigation,
  other resources, tool use and social communication.
* **EgoTextVQA**: the prompts are copied verbatim from `zhousheng97/EgoTextVQA/gpt_eval.py`.
  Answers and predictions are lower-cased and stripped of newlines first. Accuracy counts
  `"yes" in pred` over yes+no, and the score is cast to `int`. Outdoor types: location,
  description, direction, intention reasoning, others. Indoor types: hands-on, shopping,
  kitchen, book-related, gameplay, others. The model is prompted with the official
  "Answer: [Full answer]" template, the raw output goes to the judge, and the prefix is
  stripped only for ANLS/EM.
* **WearVQA**: the paper's judge prompt (NeurIPS 2025 D&B) could not be retrieved while this
  was written. `WEARVQA_JUDGE_*` in `eval/metrics/llm_judge.py` is a binary-correctness prompt
  in the same style. Replace it with the paper's prompt before reporting official numbers;
  `metrics.json` carries a note saying so. Slices: the 6 wearable image-quality issues (blur,
  cut-off, low light, unmagnified, occlusion, rotation), the 7 domains and the 10 task types,
  taken from whatever columns the release provides (`make_wearvqa.py --dry_run` prints them).

### Video protocol

Gemma 4 encodes every video frame with the same vision tower, at `video_token_budget` soft
tokens per frame (70 by default, as in `Gemma4VideoProcessor`). Frames are passed through the
processor's native video path with their real timestamps (`mm:ss` before each frame), so a
grafted student also replaces the tower for video. Frame sampling is fixed at LMDB build time:

* **EgoBlind, EgoTextVQA**: real-time questions. Frames are uniform in `[0, t_question]`, and
  the last frame is at question time.
* **EgoSchema**: 32 segment-centred frames over the 3-minute clip.

### EgoTextVQA resolution

High resolution (1920x1080) and low resolution (960x540) are separate LMDBs
(`--resize`). Gemma then resizes each frame to its token budget:

* At 70 tokens/frame (about 161K px) both settings are downsampled to the same small size and
  become almost identical.
* At 280 tokens/frame (about 645K px) the low-res frames (518K px) are slightly *upsampled*
  while the high-res frames are downsampled, so the gap is real but modest.
* The gap grows at 560 and 1120 tokens/frame (about 1.3M and 2.6M px).

Always report the per-frame budget alongside the resolution setting.
