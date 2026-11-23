"""A1 ReLU pilot: analyse finished pilot runs, then (after review) freeze thresholds.

The pilot runs themselves are trained by the normal runner:

    python scripts/a1_run.py --config configs/a1/pilot.json --workers 3

Then:

    python scripts/a1_pilot.py analyze --config configs/a1/pilot.json
        Budget check, threshold calibration and outcome summary. Writes only
        to results/a1/pilot/relu-pilot/analysis/ (git-ignored). Never trains.

    python scripts/a1_pilot.py freeze --proposed results/a1/pilot/relu-pilot/analysis/outcomes_proposed.json --note "..."
        Copies the reviewed proposal into configs/a1/outcomes.json with
        frozen=true. Refuses if that file is already frozen. Do this only
        after reviewing pilot_report.md, and before any DARL comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from darl_study.a1.outcomes import load_thresholds
from darl_study.a1.pilot import analyze_pilot, freeze_thresholds
from darl_study.a1.runner import load_experiment

REPO = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A1 ReLU pilot analysis and threshold freezing.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse finished pilot runs")
    a.add_argument("--config", type=Path, default=REPO / "configs" / "a1" / "pilot.json")
    a.add_argument("--allow-incomplete", action="store_true",
                   help="analyse the runs that are finished and list the rest (for a progress look only)")
    f = sub.add_parser("freeze", help="freeze reviewed thresholds into configs/a1/outcomes.json")
    f.add_argument("--proposed", type=Path, required=True)
    f.add_argument("--target", type=Path, default=REPO / "configs" / "a1" / "outcomes.json")
    f.add_argument("--note", required=True, help="who reviewed it and from which pilot")
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        exp = load_experiment(args.config)
        draft = load_thresholds(exp.path(exp.thresholds), purpose=exp.purpose)
        try:
            s = analyze_pilot(exp, draft, allow_incomplete=args.allow_incomplete)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        b = s["budget"]
        print(f"PILOT ANALYSIS '{exp.name}' ({s['n_runs']} runs) - not research results")
        if s["incomplete"]:
            print(f"  INCOMPLETE: {len(s['incomplete'])} run(s) missing")
        print(f"  budget: {b['largest_candidate']} epochs sufficient = {b['largest_candidate_sufficient']}; "
              f"shortest passing budget = {b['recommended_budget']}")
        top = b["budgets"][-1]
        print(f"    at {top['budget']}: {top['converged']}/{top['eligible']} non-collapsed runs converged "
              f"({top['excluded_collapsed']} collapsed excluded, {top['excluded_nonfinite']} nonfinite)")
        c = s["calibration"]
        print(f"  thresholds changed: {', '.join(c['changed']) or 'none'}; "
              f"manual review: {', '.join(c['manual_review']) or 'none'}")
        for arm, v in s["outcomes_proposed"]["arms"].items():
            print(f"  {arm:<10} n={v['n']}  " + "  ".join(
                f"{o}={x['count']}" for o, x in v["outcomes"].items()) +
                f"  born_dead={v['born_dead']['count']}")
        print(f"  report: {exp.root / 'analysis' / 'pilot_report.md'}")
        return 0

    try:
        th = freeze_thresholds(args.proposed, args.target, args.note)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"froze {args.target} (hash {th.hash()[:12]}): {th.frozen_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
