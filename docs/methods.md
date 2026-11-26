# Distillation methods from the literature

The goal is to compress Gemma 4's vision encoder while keeping the VLM's **general** vision-language abilities. Nothing
here is tuned for one application domain. The student replaces the vision tower of the unchanged Gemma 4: the
language model and the multimodal projector `embed_vision` stay frozen. So the student must produce tokens that the
original language model can read.

This page covers the four ingredients from the literature that the repository implements:

* what each one is;
* where it lives in the code;
* the study that isolates it.

The evaluation that decides between them is in [protocol.md](protocol.md), §6.

## Notation

* For an image *x*, the teacher's vision tower outputs soft tokens `T(x) ∈ R^{N×768}`. The student outputs `S(x)`,
  with the same N tokens in the same order (the student contract, protocol.md §2).
* `P = embed_vision` maps tokens into the language model's input space: a scale-free RMSNorm followed by a linear
  map (768 → 1536 for E2B).
* `P` reads only the **direction** of each token. RMSNorm without a learned scale removes each token's norm.
* Intermediate tokens: `T_l(x)` is teacher layer *l* and `S_j(x)` is student layer *j*. Both are average-pooled to the
  same 48-px token grid.

## (1) Function matching on the output tokens (Beyer et al., CVPR 2022)

*Knowledge distillation: a good teacher is patient and consistent.* The student reproduces the teacher's function on
the same input. Beyer et al. identify three ingredients:

* the teacher and the student see **exactly the same** augmented image ("consistent");
* aggressive **mixup** widens the input distribution on which the functions are matched;
* training is **long** ("patient").

The loss here is the main objective of the repository: function matching on the tokens the language model reads.

```
L_FM = MSE(P(S(x̃)), P(T(x̃))) + (1 − cos(P(S(x̃)), P(T(x̃))))   (mean over tokens)
x̃ = λ·x_i + (1−λ)·x_j,   λ ~ U(0, 1) = Beta(1, 1),   the same x̃ for both models
```

| piece | code |
|---|---|
| consistent views | `DualViewCollator`: one geometric transform per image, applied to both views |
| mixup | `data_module.collator.mixup_alpha=1.0` (`DualViewCollator._mixup`): images of a shape-exact batch are mixed before the teacher and student views are made, so both see identical pixels |
| loss | `config/objective/fm.yaml` (`token_feature` mse + cosine, after `P`) |
| patience | `trainer.max_epochs` at fixed images seen; the learning curves of every run (`TrajectoryCallback`) show whether a run has converged |

Validation and test images are never mixed (`val_collator: {train: false}`).

## (2) Manifold distillation on the output tokens (Hao et al., NeurIPS 2022)

*Learning efficient vision transformers via fine-grained manifold distillation.* The student reproduces the teacher's
**relations between tokens** instead of the tokens themselves. Tokens are L2-normalised, and relation matrices are
`M(F) = F F^T`.

| term | relation matrix | what it captures |
|---|---|---|
| intra-image | `N×N` per image | spatial structure inside one image |
| inter-image | `B×B` per token position (shape-exact batches: equal N) | how images relate at the same position |
| random | `K×K` over K tokens sampled from the whole batch (K = 256) | a cheap mix of both |

```
L_MD = α·mean_i ||M(S_i) − M(T_i)||² + β·mean_p ||M(S_{:,p}) − M(T_{:,p})||² + γ·||M(S_K) − M(T_K)||²
```

The relations are invariant to a rotation of the feature space and to per-token scale. So the relation terms alone
fix the student's tokens only up to one orthogonal map. Two facts make the relation-only objective (`md`)
meaningful here:

* `P` ignores token norms;
* the student starts from the teacher's weights (`student.init=teacher`).

The question the study answers is whether relations keep the tokens readable by the frozen language model: `md`
compared with `fm_md` and `fm`.

| piece | code |
|---|---|
| loss | `ManifoldLoss` (`onboard_vlm/distill/losses.py`, `name: manifold`) |
| objectives | `config/objective/md.yaml` (relations only), `fm_md.yaml` (with FM), measured after `P` |
| weights | `md.scale × (md.intra, md.inter, md.random)`. Each term is its own entry, so it gets its own training curve. |
| reduction | `md.reduction`: `mean` over matrix entries (default, independent of N) or `sum`, the squared Frobenius norm per matrix |

**Setting the weights (do this before the objective study).** The terms are small in absolute value (cosine
relations of nearby features), so their scale relative to `L_FM` is a real hyperparameter.

* `--study md_scale` trains FM + MD with `md.scale ∈ {1, 10, 100, 1000}` on one seed.
* The scale is chosen on **held-out** data only: E1 held-out fidelity and the E2 ImageNet k-NN. The E3 benchmarks
  are never used for choosing.
* The chosen value is then fixed for every other run (`--md-scale`).

α, β and γ default to 1. Set them to the values of the paper's implementation details if you follow it exactly:
check the exact numbers and the reduction in the paper (they could not be verified from this environment).

## (3) Applying (1) or (2) to intermediate layers

`layer_map: auto` pairs 3 student layers, spread over the student's depth with the last included, with the teacher
layers at the same relative depth.

| objective | intermediate term | code |
|---|---|---|
| `fm_fmL` | FitNets-style feature matching (Romero et al., ICLR 2015): a training-only linear head maps student layer *j* to the teacher width, LayerNorm-normalised MSE to `T_l` | `intermediate_layer`, `mode: feature` |
| `fm_mdL` | manifold distillation on hidden layers, Hao et al.'s own setting: relations need no head, the student's own width is used | `intermediate_layer`, `mode: manifold`, `task.adapter.layer_heads=false` |

Both keep `L_FM` on the output tokens, since the language model reads the output. The heads are discarded at export.

## (4) Weight multiplexing (Zhang et al., CVPR 2022)

*MiniViT: compressing vision transformers with weight multiplexing.* Consecutive layers share their weights, up to
one small layer-specific transformation. In `gemma4_shrink` (`student.share_every = k`):

* every block of `k` consecutive layers uses one set of attention and MLP weights, those of the block's first layer;
* each layer keeps its own norms (cheap and unshared, as in MiniViT);
* each layer applies its own **identity-initialised** transformation to the shared block's output (`student.share_transform`):

| `share_transform` | layer-specific transformation | MiniViT counterpart |
|---|---|---|
| `linear` | `h ↦ W h + b`, with `W = I`, `b = 0` at init: one `D×D` linear layer | the per-layer linear transformation |
| `dwconv` | 3×3 depthwise convolution on the image's patch grid (padding patches excluded), identity at init | the per-layer depthwise-conv transformation of the MLP |
| `none` | none: pure weight sharing | the "w/o transformation" ablation |

Two properties of the student:

* With `init=teacher`, the shared attention and MLP of a block are those of the teacher layer selected for the block's
  first layer; each layer's norms come from the teacher layer selected for that layer. At step 0 the transformations
  are the identity, so a shared student starts as a teacher-derived network with repeated blocks.
* Parameters are counted **once per shared tensor**: `student_stats.json` holds `parameters` and
  `parameters_with_repeats`. This count drives the params axis of every plot.

The multiplex study separates the two effects of sharing:

* same unique weights, more depth: 8 layers unshared vs 16 layers shared in pairs (none / linear / dwconv);
* same depth, fewer weights: 8 layers unshared vs 8 shared in pairs vs 4 unshared.

## E2B and E4B: one vision tower, two targets

The four Gemma 4 checkpoints (E2B, E4B, base and `-it`) share the vision tower byte for byte (658 tensors, checked
2026-11-23; `onboard_vlm/teachers/gemma4.py`). What differs is `embed_vision` (768 → 1536 for E2B, 768 → 2560 for
E4B) and the language model that reads the result. So "distilling E2B's and E4B's ViT" means distilling one function
into one of three targets:

* **vision space** (`objective=fmV`): the 768-d tokens themselves, RMS-normalised MSE + cosine. One student for
  both VLMs.
* **E2B's LM-input space** (`teacher=gemma4_e2b objective=fm`): what E2B's language model reads.
* **E4B's LM-input space** (`teacher=gemma4_e4b objective=fm`).

Every student is then evaluated in **both** VLMs. That gives a 3 × 2 table: does matching in one LM's input space
over-fit to that LM, and does one student serve both? The teacher key is part of every run name
(`lit_e2b_it_…`, `lit_e4b_it_…`). Evaluations in the other VLM go to `eval_vlm_<key>/` of each export, and
`plot_study.py --vlm <key>` reads them.

## Building up: the order of experiments

Each step adds one ingredient to the previous one at equal data and compute, so every gain is attributable. Steps
3–5 use the target that step 2 selects. If both LM spaces win in their own VLM, run steps 3–5 for E2B first (the
cheaper VLM to evaluate) and repeat the winning recipe for E4B.

| step | study | runs (seeds) | question | needs |
|---|---|---|---|---|
| 0 | `references` | teacher in E2B and E4B: general suite, blind floor, E2 probes | the ceiling and the floor of every comparison | none |
| 1 | `identity --seeds 0` | 16 × 768 from the teacher (the teacher itself) | does training keep the teacher exact (fidelity 1 at every point)? A pipeline check | 0 |
| 2 | `target` | output matching without mixup, into vision / E2B / E4B space (3 × 3) | the baseline and where to match | 1 |
| 3 | `objective --only '*_fm_mx*'` | + consistent mixup (2 × 3; the no-mixup run is step 2's) | (1) Beyer et al.: does function matching help? | 2 |
| 4a | `md_scale --seeds 0` | + relations, 4 scales (4 × 1) | how much weight relations get (held-out selection) | 3 |
| 4b | `objective --md-scale <s>` | + relations (MD, FM+MD), + intermediate layers (FM@L, MD@L) | (2), (3) Hao et al., FitNets | 4a |
| 5 | `multiplex --objective <best>` | weight sharing, 6 students | (4) MiniViT: depth for free at fixed parameters? | 4b |
| 6 | `depth`, `width`, `grid` | the size axis with the final recipe | how small can the student get? | 5 |

Step 0 of a teacher-initialised student is the untrained, truncated and narrowed teacher. `--lmms-steps first,last`
evaluates that "no training" floor on the general suite too (once is enough: it does not depend on the seed).

```bash
source scripts/paths.sh
S=scripts/slurm/submit_shrink_study.sh
bash $S --account <project> --study references --eval-vlms e2b,e4b
bash $S --account <project> --study identity --seeds 0
bash $S --account <project> --study target
bash $S --account <project> --study objective --only '*_fm_mx*'
bash $S --account <project> --study md_scale --seeds 0
bash $S --account <project> --study objective --md-scale <chosen>
bash $S --account <project> --study multiplex --objective <best>
```

Add `--teacher gemma4_e4b` to repeat a study in E4B's space and `--eval-vlms e2b,e4b` to evaluate its students in
both VLMs. `--dry-run` prints what would be submitted, and `--list` prints the run names and their overrides.

Every literature run uses `+experiment=lit/base`:

* the same student (`--student "8 384"`, teacher-initialised);
* the same data (`general_vision_lite`, general-purpose);
* the same images seen (5M), schedule and evaluation;
* one factor changes per comparison.

Run names spell out every factor:
`lit_<teacher key>_<objective>[_w<md scale>]_mx<mixup alpha>_L<layers>_D<width>_x<k>-<transform>`.

| study | runs | question |
|---|---|---|
| `target` | FM w/o mixup into vision / E2B / E4B space, each evaluated in both VLMs | where to match, and does one student serve both |
| `md_scale` | FM+MD, scale ∈ {1, 10, 100, 1000} | how strongly to weight relations (held-out selection) |
| `objective` | FM w/o mixup, FM, MD, FM+MD, FM+FM@layers, FM+MD@layers | which of (1), (2), (3) keeps the most general ability; does mixup matter |
| `multiplex` | 8×1, 16×2 {none, linear, dwconv}, 8×2 linear, 4×1 | does sharing with a transformation buy depth at a fixed parameter count (4) |

Each training job is followed by `scripts/trajectory_eval.py`:

* E1, E2 and the E3 mini-benchmarks on every export;
* the full general suite on the last export;
* with `--eval-vlms`, a second job for E3 and the suite in the other VLM.

Each teacher is evaluated once in its own VLM, together with the blind control. The launcher prints the plot and
table commands per VLM:

* the plots stay at four series per figure;
* the tables give seed mean ± std, retention, and visual retention against that VLM's blind floor.

## References

* L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, A. Kolesnikov. Knowledge distillation: a good teacher is patient
  and consistent. CVPR 2022.
* Z. Hao, J. Guo, D. Jia, K. Han, Y. Tang, C. Zhang, H. Hu, Y. Wang. Learning efficient vision transformers via
  fine-grained manifold distillation. NeurIPS 2022.
* J. Zhang, H. Peng, K. Wu, M. Liu, B. Xiao, J. Fu, L. Yuan. MiniViT: compressing vision transformers with weight
  multiplexing. CVPR 2022.
* A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, Y. Bengio. FitNets: hints for thin deep nets. ICLR 2015.
* S. Tong, E. Brown, P. Wu, et al. Cambrian-1: a fully open, vision-centric exploration of multimodal LLMs.
  NeurIPS 2024 (the evaluation protocol).
