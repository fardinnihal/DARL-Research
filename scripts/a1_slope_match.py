"""A1 Step 9: compute the slope-matched LeakyReLU value and write configs/a1/main.json.

Run after the DARL calibration has finished:

    python scripts/a1_run.py --config configs/a1/calibration.json --workers 3
    python scripts/a1_slope_match.py            # preview: prints the slopes, writes nothing
    python scripts/a1_slope_match.py --write    # writes configs/a1/main.json and its record

The slope is DARL's mean negative slope over all hidden units and all training
steps, averaged over the completed calibration runs of each initialization
(decision D7). It uses training-time DARL state only. It refuses unfinished
calibration runs, a calibration that is not DARL with the approved defaults,
main seeds that overlap the pilot or the calibration, draft thresholds, and an
existing main.json with different content (never overwritten). Nothing trains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from darl_study.a1.checkpoint import atomic_write_text
from darl_study.a1.main_plan import build_main_config

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "a1"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Slope-matched LeakyReLU value and the A1 main config.")
    ap.add_argument("--calibration", type=Path, default=CFG / "calibration.json")
    ap.add_argument("--template", type=Path, default=CFG / "main_template.json")
    ap.add_argument("--pilot", type=Path, default=CFG / "pilot.json", help="its seeds must not be reused")
    ap.add_argument("--out", type=Path, default=CFG / "main.json")
    ap.add_argument("--write", action="store_true", help="write --out and its slope_match record")
    args = ap.parse_args(argv)
    try:
        main_dict, match = build_main_config(args.calibration, args.template, args.pilot)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"SLOPE MATCH from calibration '{match['calibration_experiment']}' "
          f"(hash {match['calibration_experiment_hash'][:12]}), seeds "
          f"{match['calibration_seeds']['start']}-"
          f"{match['calibration_seeds']['start'] + match['calibration_seeds']['count'] - 1}")
    print(f"  rule: {match['rule']}")
    for init, v in sorted(match["per_init"].items()):
        s = v["time_mean_stats"]
        f = v["final_mean_stats_reported_only"]
        print(f"  {init:<4} alpha_bar = {v['alpha_bar']:.6f}  (n={s['n']}, sd {s['sd'] or 0:.6f}, "
              f"range {s['min']:.6f}-{s['max']:.6f}; nonfinite excluded {v['n_nonfinite_excluded']}; "
              f"final-alpha mean {f['mean'] if f['mean'] is None else round(f['mean'], 6)}, reported only)")
        print(f"       calibration outcomes: {v['calibration_outcomes']}")
    for a in main_dict["arms"]:
        print(f"  arm {a['name']:<13} {a['activation']:<10} {a['init']:<4} {a.get('act_kwargs', {})}")
    print(f"  main: {main_dict['name']}, seeds {main_dict['seeds']['start']}-"
          f"{main_dict['seeds']['start'] + main_dict['seeds']['count'] - 1}, "
          f"experiment hash {match['main_experiment_hash'][:12]}, thresholds {match['main_thresholds_hash'][:12]}")

    text = json.dumps(main_dict, indent=2) + "\n"
    record = args.out.with_name(args.out.stem + "_slope_match.json")
    if args.out.exists():
        old = json.loads(args.out.read_text(encoding="utf-8"))
        same = {k: v for k, v in old.items() if k != "comment"} == {k: v for k, v in main_dict.items()
                                                                     if k != "comment"}
        if not same:
            print(f"error: {args.out} exists with different content; it is never overwritten. "
                  "Remove it only if no main run has started.", file=sys.stderr)
            return 2
        print(f"{args.out} already holds exactly this config; nothing to do")
        return 0
    if not args.write:
        print("PREVIEW: nothing written. Rerun with --write to create the main config.")
        return 0
    atomic_write_text(record, json.dumps(match, indent=2) + "\n")
    atomic_write_text(args.out, text)
    print(f"wrote {args.out} and {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
