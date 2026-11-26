"""Evaluate every student export of a run's trajectory (``TrajectoryCallback``), for curves over training.

For each ``<run_dir>/step_<n>/`` (in step order), on fixed subsets so every point is measured on the same items:

* E1: fidelity on every held-out / corpus / target set (``max_images_per_set`` images each, a few budgets);
* E2: ImageNet-1k k-NN and linear probes (``train_per_class`` per class, ``val_images`` val) and NYUv2 depth;
* E3: downstream with the student grafted into Gemma 4, rule-based scores only (no LLM judge): TextVQA val and
  CV-Bench (a seeded sample of ``e3_items`` questions each; ``--e3`` also accepts vizwiz_vqa, egoschema_subset);
* lmms (opt-in, ``--evals ...,lmms``): the full general suite (``onboard_vlm/eval/suites.py``, Cambrian-1 protocol,
  lmms-eval, every item) on the exports of ``--lmms_steps`` (default: the first and the last, i.e. before and after
  training), one job per category; with ``--teacher`` also the *blind* control (grey images).

Outputs go to ``<run_dir>/step_<n>/eval/{e1,e2,e3/<benchmark>,lmms/<category>}``; finished evaluations are skipped, so the
script can be re-run. Jobs run in parallel, one per GPU. ``--teacher`` evaluates the original encoder once
(E2, E3) into ``<run_dir>`` as the reference line of the plots.

    python scripts/trajectory_eval.py --run_dir $ONBOARD_VLM_OUT/trajectories/<run_id> --gpus 4
    python scripts/trajectory_eval.py --run_dir $ONBOARD_VLM_OUT/trajectories/_teacher_e2b --teacher --gpus 4
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E3_BENCHMARKS = ("textvqa", "cvbench")  # general (OCR, vision-centric); the application sets stay available via --e3


def jobs_for(target: str, student_dir: str | None, a) -> list[tuple[str, list[str], str]]:
    """``(name, command, done_file)`` for one export (``student_dir``) or the teacher (``None``)."""
    out = os.path.join(target, "eval")
    tag = os.path.basename(os.path.normpath(target))
    run = os.path.basename(os.path.dirname(os.path.normpath(target))) if student_dir else tag
    common = [f"teacher={a.teacher_config}", *a.extra]
    jobs = []
    if "e1" in a.evals and student_dir:
        d = os.path.join(out, "e1")
        jobs.append(("e1", [sys.executable, "-u", "run_fidelity.py", f"run_name={run}/{tag}", f"student_dir={student_dir}",
                            f"output_dir={d}", f"hydra.run.dir={d}/hydra", "allow_missing_sources=true",
                            f"max_images_per_set={a.e1_images}", f"budgets=[{a.budgets}]", f"bootstrap.n={a.bootstrap}", *common],
                     os.path.join(d, "summary.json")))
    if "e2" in a.evals:
        d = os.path.join(out, "e2")
        enc = [f"encoder.student_dir={student_dir}"] if student_dir else ["encoder.teacher=true"]
        jobs.append(("e2", [sys.executable, "-u", "run_probe.py", f"run_name={run}/{tag}", *enc, f"tasks=[{a.probes}]",
                            f"imagenet.train_per_class={a.train_per_class}", f"imagenet.val_limit={a.val_images}",
                            f"output_dir={d}", f"hydra.run.dir={d}/hydra", f"bootstrap.n_boot={a.bootstrap}", *common],
                     os.path.join(d, "results.json")))
    if "e3" in a.evals:
        for b in a.e3.split(","):
            d = os.path.join(out, "e3", b)
            stu = [f"student_dir={student_dir}"] if student_dir else []
            jobs.append((f"e3/{b}", [sys.executable, "-u", "run_eval.py", f"benchmark={b}", f"run_name={run}/{tag}", *stu,
                                     f"output_dir={d}", f"hydra.run.dir={d}/hydra", "use_judge=false",
                                     f"++benchmark.dataset.sample={a.e3_items}", f"vlm_id={a.vlm_id}", *a.extra],
                         os.path.join(d, "metrics.json")))
    if "lmms" in a.evals and (student_dir is None or tag in a.lmms_tags):
        sys.path.insert(0, ROOT)
        from onboard_vlm.eval.suites import CATEGORIES, EXTRA, JUDGED

        model_args = f"pretrained={a.vlm_id},max_soft_tokens={a.lmms_budget}" + (f",student_dir={student_dir}" if student_dir else "")
        for blind in ([False, True] if student_dir is None else [False]):
            for cat, ts in {**CATEGORIES, **EXTRA}.items():
                ts = [t for t in ts if a.judged or t not in JUDGED]
                if not ts:
                    continue
                slug = re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")
                d = os.path.join(out, "lmms_blind" if blind else "lmms", slug)
                jobs.append((f"{'lmms_blind' if blind else 'lmms'}/{slug}",
                             [sys.executable, "-u", "run_lmms_eval.py", "--model", "onboard_gemma4",
                              "--model_args", model_args + (",blind=True" if blind else ""), "--tasks", ",".join(ts),
                              "--batch_size", "1", "--log_samples", "--output_path", d],
                             os.path.join(d, "**", "*_results.json")))
    return jobs


def done(marker: str) -> bool:
    return bool(glob.glob(marker, recursive=True)) if "*" in marker else os.path.isfile(marker)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="<output_root>/trajectories/<run_id> (or a new directory with --teacher)")
    p.add_argument("--teacher", dest="teacher_mode", action="store_true", help="evaluate the original Gemma 4 encoder once (reference lines)")
    p.add_argument("--evals", default="e1,e2,e3")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--teacher_config", default="gemma4_e2b", help="config/teacher/<name>.yaml of the run (gemma4_e2b | gemma4_e4b)")
    p.add_argument("--vlm_id", default=None, help="deployment VLM for E3 (default: the teacher's)")
    p.add_argument("--budgets", default="70,280,1120")
    p.add_argument("--e1_images", type=int, default=500)
    p.add_argument("--probes", default="imagenet,nyuv2")
    p.add_argument("--train_per_class", type=int, default=10)
    p.add_argument("--val_images", type=int, default=10000)
    p.add_argument("--e3", default=",".join(E3_BENCHMARKS))
    p.add_argument("--e3_items", type=int, default=500)
    p.add_argument("--lmms_steps", default="first,last", help="exports that get the full general suite: first,last | all | step numbers")
    p.add_argument("--lmms_budget", type=int, default=280)
    p.add_argument("--judged", action="store_true", help="also the GPT-judged benchmarks of the suite (needs OPENAI_API_KEY)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--steps", default=None, help="only these steps (comma-separated), e.g. 0,1000")
    p.add_argument("--extra", nargs="*", default=[], help="Hydra overrides appended to every command (e.g. device=cpu)")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    a.vlm_id = a.vlm_id or {"gemma4_e2b": "google/gemma-4-E2B-it", "gemma4_e4b": "google/gemma-4-E4B-it"}.get(a.teacher_config, "google/gemma-4-E2B-it")

    if a.teacher_mode:
        os.makedirs(a.run_dir, exist_ok=True)
        a.evals = ",".join(e for e in a.evals.split(",") if e != "e1")  # E1 of the teacher is 1 by definition
        targets = [(a.run_dir, None)]
    else:
        steps = sorted(glob.glob(os.path.join(a.run_dir, "step_*")))
        if a.steps:
            want = {int(s) for s in a.steps.split(",")}
            steps = [s for s in steps if int(os.path.basename(s)[5:]) in want]
        if not steps:
            sys.exit(f"no step_* exports under {a.run_dir} (TrajectoryCallback)")
        targets = [(s, s) for s in steps]
    names = [os.path.basename(os.path.normpath(t)) for t, _ in targets]
    if a.lmms_steps == "all":
        a.lmms_tags = set(names)
    else:
        want = set(a.lmms_steps.split(","))
        a.lmms_tags = {n for i, n in enumerate(names) if (i == 0 and "first" in want) or (i == len(names) - 1 and "last" in want)
                       or n[5:].lstrip("0") in {w.lstrip("0") for w in want}}
    todo, n_done = [], 0
    for target, student in targets:
        for name, cmd, marker in jobs_for(target, student, a):
            if done(marker):
                n_done += 1
            else:
                todo.append((target, name, cmd))
    print(f"[trajectory_eval] {a.run_dir}: {len(todo)} evaluations to run, {n_done} already done, on {a.gpus} GPU(s)", flush=True)
    if a.dry_run:
        for target, name, cmd in todo:
            print(f"  {os.path.basename(target)} {name}: {' '.join(cmd[2:])}")
        return

    gpus: queue.Queue[int] = queue.Queue()
    for g in range(a.gpus):
        gpus.put(g)
    failures: list[str] = []

    def work(target: str, name: str, cmd: list[str]) -> None:
        g = gpus.get()
        try:
            logf = os.path.join(target, "eval", f"{name.replace('/', '_')}.log")
            os.makedirs(os.path.dirname(logf), exist_ok=True)
            with open(logf, "w") as f:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}).returncode
            print(f"[trajectory_eval] {os.path.basename(target)} {name}: {'ok' if rc == 0 else f'FAILED (exit {rc}, see {logf})'}", flush=True)
            if rc:
                failures.append(f"{target} {name}")
        finally:
            gpus.put(g)

    threads = [threading.Thread(target=work, args=job) for job in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        sys.exit(f"[trajectory_eval] {len(failures)} evaluation(s) failed: {failures}")


if __name__ == "__main__":
    main()
