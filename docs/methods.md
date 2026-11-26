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

## The studies

Every literature run uses `+experiment=lit/base`:

* the same student (`--student "8 384"`, teacher-initialised);
* the same data (`general_vision_lite`, general-purpose);
* the same images seen (5M), schedule and evaluation;
* one factor changes per comparison.

Run names spell out every factor: `lit_<objective>[_w<md scale>]_mx<mixup alpha>_L<layers>_D<width>_x<k>-<transform>`.

```bash
source scripts/paths.sh
bash scripts/slurm/submit_shrink_study.sh --account <project> --study md_scale --seeds 0       # first: the MD weight scale
bash scripts/slurm/submit_shrink_study.sh --account <project> --study objective --md-scale <chosen>
bash scripts/slurm/submit_shrink_study.sh --account <project> --study multiplex --objective <best objective>
```

| study | runs | question |
|---|---|---|
| `md_scale` | FM+MD, scale ∈ {1, 10, 100, 1000} | how strongly to weight relations (held-out selection) |
| `objective` | FM, FM w/o mixup, MD, FM+MD, FM+FM@layers, FM+MD@layers | which of (1), (2), (3) keeps the most general ability; does mixup matter |
| `multiplex` | 8×1, 16×2 {none, linear, dwconv}, 8×2 linear, 4×1 | does sharing with a transformation buy depth at a fixed parameter count (4) |

Each training job is followed by `scripts/trajectory_eval.py`:

* E1, E2 and the E3 mini-benchmarks on every export;
* the full general suite on the last export.

The teacher is evaluated once, together with the blind control. The launcher prints the plot commands and the
table command:

* the plots stay at four series per figure;
* the table gives seed mean ± std, retention, and visual retention against the blind floor.

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
