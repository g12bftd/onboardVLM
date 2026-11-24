"""Select a scenario-stratified subset of Ego4D videos for the distillation corpus.

Reads ``ego4d.json`` (``videos[*].video_uid / duration_sec / scenarios``), drops every video listed
in the benchmark blocklist (EgoSchema / EgoTextVQA-Indoor sources), then fills ``--hours`` by
round-robin over scenarios (random order inside each scenario, seeded). Round-robin caps the
dominant scenarios (cooking, cleaning, ...) so the subset covers the long tail of activities and
places; the selected uids, hours per scenario and the seed are written next to the uid list.

Example:
  python make_datasets/download/corpus/select_ego4d_videos.py --metadata $RAW_ROOT/ego4d/ego4d.json \\
      --blocklist $CORPUS_ROOT/eval_blocklist.json --hours 1000 --output $RAW_ROOT/ego4d/corpus_video_uids.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--blocklist", required=True, help="eval_blocklist.json (video_ids of benchmark sources)")
    p.add_argument("--hours", type=float, default=1000.0)
    p.add_argument("--min_duration_sec", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with open(args.metadata) as f:
        videos = json.load(f)["videos"]
    with open(args.blocklist) as f:
        blocked = set(json.load(f).get("video_ids", []))
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    n_blocked = 0
    for v in videos:
        if v["video_uid"] in blocked:
            n_blocked += 1
            continue
        if float(v.get("duration_sec") or 0) < args.min_duration_sec:
            continue
        scen = sorted(v.get("scenarios") or ["unknown"])[0]
        by_scenario[scen].append(v)
    rng = random.Random(args.seed)
    queues = {k: rng.sample(vs, len(vs)) for k, vs in sorted(by_scenario.items())}
    chosen, hours, per_scen = [], 0.0, defaultdict(float)
    while hours < args.hours and any(queues.values()):
        for k in list(queues):
            if not queues[k] or hours >= args.hours:
                continue
            v = queues[k].pop()
            chosen.append(v["video_uid"])
            h = float(v["duration_sec"]) / 3600
            hours += h
            per_scen[k] += h
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(chosen) + "\n")
    summary = {"videos": len(chosen), "hours": round(hours, 1), "blocked_benchmark_videos": n_blocked, "seed": args.seed,
               "hours_per_scenario": {k: round(v, 1) for k, v in sorted(per_scen.items(), key=lambda kv: -kv[1])}}
    with open(os.path.splitext(args.output)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"selected {len(chosen)} videos / {hours:.0f} h from {len(by_scenario)} scenarios ({n_blocked} benchmark videos excluded)")


if __name__ == "__main__":
    main()
