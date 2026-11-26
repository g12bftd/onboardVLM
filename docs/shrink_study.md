# Shrinking study: what distillation does, size by size, over training time

The first systematic look at the distillation: take Gemma 4's own vision encoder (16 layers x 768, 12 heads,
~167M parameters with its 2 x 10,240 x 768 position table), make it smaller along one axis at a time,
distil it with the main objective, and measure everything over training time.

## 1. The student: the teacher, made smaller (`gemma4_shrink`)

* `transformers`' `Gemma4VisionModel`, unchanged, built from a copy of the **teacher's own** `vision_config`.
  Only depth, width, head count and MLP width change; the patch embedder, 2-D RoPE, position table, clipped
  linears, norms, GELU and the pooler are the teacher's.
* At full size with `init: teacher` it is the teacher, weight for weight (tested: max |difference| = 0). This
  run, the **identity anchor**, must start at fidelity 1 and stay there; if it does not, the recipe is
  broken, not the student.
* Initialisation (`student.init`):
  * `teacher`: depth by layer selection (`uniform`: evenly spaced layers, as DistilBERT; `first`; `last`);
    width by weight selection (Xu et al., ICLR 2024), the same residual channels in every tensor, whole heads
    and MLP units. The output head puts the kept channels back at their teacher positions (x sqrt(768 / width),
    undoing the pooler's sqrt(width)).
  * `random`: transformers' initialiser, seeded by `init_seed`.
* `layer_map: auto`: intermediate supervision of 3 student layers spread over its depth (the last included),
  each against the teacher layer it was initialised from (or at the same relative depth).

## 2. Studies (`scripts/slurm/submit_shrink_study.sh --study ...`)

Every run uses `+experiment=shrink/base`: the main objective (LM-input-space cosine + 3 intermediate
layers), the same corpus, budgets, schedule and compute (10 epochs x 500k = 5M image samples). One factor
changes per study.

| study | configurations | question |
|---|---|---|
| `identity` | 16 x 768, teacher init | is the recipe sound (fidelity stays at 1)? |
| `depth` | {12, 8, 6, 4, 2} x 768, teacher init | how does fidelity degrade with depth? |
| `width` | 16 x {512, 384, 256, 128}, teacher init | with width? |
| `init` | 8 x 384: teacher-uniform, teacher-first, random | does inheriting the teacher's weights matter, and how? |
| `grid` | {4, 8, 12} x {256, 384, 512}, teacher init | depth vs width at equal parameters |

The same launcher runs the literature studies (`--study md_scale | objective | multiplex`: function matching,
manifold distillation, intermediate layers, weight multiplexing), described in [methods.md](methods.md).

Run `identity` and `depth` first; `grid` (27 runs with 3 seeds) after the first results.

## 3. What is measured, and when

| when | what | where |
|---|---|---|
| step 0 and every half epoch (val split: corpus groups, budgets 70 / 280 / 1120) | LM-space cosine, MSE, relative error; vision-space cosine, MSE, relative error; worst-5% token cosine; CKA; per budget and per group; training losses | `logs/onboard_vlm/<run_id>/metrics.csv` |
| 10 checkpoints at 0, 1, 2, 5, 10, 20, 35, 50, 75, 100% of training (student exports) | E1 fidelity on fixed subsets (500 images per held-out set, budgets 70 / 280 / 1120); E2 ImageNet k-NN and linear probes, NYUv2 depth; E3 with the student grafted into Gemma 4 E2B, rule-based scores, a fixed sample of 500 questions each: TextVQA val, CV-Bench; on the last export, the full general suite (Cambrian-1 protocol, [protocol.md](protocol.md) E3) | `trajectories/<run_id>/step_*/eval/` |
| once | the teacher's E2 / E3 on the same items (reference lines), and the blind control of the general suite | `trajectories/_teacher_e2b/eval/` |

Every curve has a compute axis (images and soft tokens seen), and every checkpoint is evaluated on the same items.

## 4. Randomness

| source | controlled by | in the study |
|---|---|---|
| student initialisation | `init_seed` (gemma4_shrink) | none for teacher init (deterministic), seeded for random init |
| data: sample order, crops, aspect ratios, budgets per batch | `data_seed` | varies with the seed |
| new heads (intermediate-layer projections) | `seed` | varies with the seed |
| GPU kernels (atomic adds in attention backward) | not controllable at speed | present; covered by the seeds |
| evaluation subsets, bootstrap resamples | fixed seeds | identical for every model and checkpoint |

* `init_seed` and `data_seed` default to `seed`. Three seeds per configuration. To split the variance,
  vary one with the other fixed (e.g. `seed=0 init_seed=1,2`); run ids record them (`_s0_i1_d0`).
* Plots show every seed (thin) and the seed mean (bold). Final values are seed mean +- std.
* Comparisons between configurations (`scripts/aggregate_results.py`) resample seeds and items together,
  with Holm correction across sets. The 500-item E3 subsets have a 95% CI of about +-4 points: they show
  trends over training, not final numbers (run the full sets on the final checkpoints).
* The hyperparameters are fixed across sizes, not tuned per size. Before claiming a size trend, check the
  learning rate at the two extremes (e.g. `task.optimizer.lr=5e-4,2e-3` on 2 x 768 and 16 x 768).

## 5. Commands

```bash
source scripts/paths.sh
bash scripts/slurm/submit_shrink_study.sh --account <project> --study identity --dry-run -- allow_missing_sources=true
bash scripts/slurm/submit_shrink_study.sh --account <project> --study identity -- allow_missing_sources=true
bash scripts/slurm/submit_shrink_study.sh --account <project> --study depth -- allow_missing_sources=true
```

`allow_missing_sources=true` only while the corpus is preliminary ([preliminary.md](preliminary.md)). The
script prints the plot command; `scripts/plot_study.py` writes PNG, PDF and a CSV table per figure:
`curves_fidelity`, `curves_budgets`, `curves_groups`, `trajectory_e1/e2/e3`, `final_vs_params`,
`grid_<metric>` (grid study) and `fidelity_vs_downstream` (Spearman rho, 95% CI from a bootstrap over runs).

## 6. Reading the results

1. `identity`: LM-space cosine 1.000 at step 0, and still ~1 at the end. A drop means the objective or
   optimiser moves a perfect student: fix the recipe before anything else.
2. Step 0 of `depth` / `width` (teacher init): how much the inherited network keeps before any training.
3. The curves: how fast each size closes the gap and where it plateaus; per budget (does 1,120 lag?) and per
   group (text-rich vs web).
4. `final_vs_params`: depth vs width at equal parameters; E2 / E3 against the teacher's dashed line.
5. `fidelity_vs_downstream`: whether LM-space fidelity predicts downstream accuracy, which decides whether
   E1 can stand in for E3 when selecting students.
