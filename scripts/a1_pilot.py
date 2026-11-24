"""A1 ReLU pilot: analyse finished pilot runs, then (after review) freeze thresholds.

The pilot runs themselves are trained by the normal runner:

    python scripts/a1_run.py --config configs/a1/pilot.json --workers 3

Then:

    python scripts/a1_pilot.py analyze --config configs/a1/pilot.json
        Budget check (rule v2, loss-floor aware, with sensitivity to the floor),
        threshold calibration and outcome summary. Writes only to
        results/a1/pilot/relu-pilot/analysis_budget-v2/ (git-ignored); the
        original rule-v1 reports in analysis/ are never touched. Never trains.

    python scripts/a1_pilot.py preview --proposed results/a1/pilot/relu-pilot/analysis_budget-v2/outcomes_proposed.json --expected-hash <hash>
        Shows exactly what freeze would write. Writes nothing.

    python scripts/a1_pilot.py freeze --proposed ...outcomes_proposed.json --expected-hash <hash> --note "..."
        Copies the reviewed proposal into configs/a1/outcomes.json with
        frozen=true. Refuses unless the proposal's hash matches the reviewed
        one, success_val_mse is the predetermined 0.001, and the target is not
        already frozen. Do this only after reviewing pilot_report.md, and
        before any DARL comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from darl_study.a1.outcomes import load_thresholds
from darl_study.a1.pilot import ANALYSIS_DIRNAME, analyze_pilot, freeze_thresholds, preview_freeze
from darl_study.a1.runner import load_experiment

REPO = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A1 ReLU pilot analysis and threshold freezing.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse finished pilot runs")
    a.add_argument("--config", type=Path, default=REPO / "configs" / "a1" / "pilot.json")
    a.add_argument("--allow-incomplete", action="store_true",
                   help="analyse the runs that are finished and list the rest (for a progress look only)")
    pv = sub.add_parser("preview", help="show exactly what freeze would write; writes nothing")
    f = sub.add_parser("freeze", help="freeze reviewed thresholds into configs/a1/outcomes.json")
    for p in (pv, f):
        p.add_argument("--proposed", type=Path, required=True)
        p.add_argument("--target", type=Path, default=REPO / "configs" / "a1" / "outcomes.json")
        p.add_argument("--expected-hash", required=True,
                       help="proposed hash from the reviewed pilot_report.md (at least 12 hex characters)")
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
        print(f"  budget rule {b['criterion']['rule_version']}: stable if {b['criterion']['stable_if']}")
        print(f"  budget: {b['largest_candidate']} epochs sufficient = {b['largest_candidate_sufficient']}; "
              f"shortest passing budget = {b['recommended_budget']}")
        top = b["budgets"][-1]
        print(f"    at {top['budget']}: {top['converged']}/{top['eligible']} non-collapsed runs budget-stable "
              f"({top['met_by_change']} by change, {top['met_by_floor_only']} by floor only; "
              f"{top['excluded_collapsed']} collapsed excluded, {top['excluded_nonfinite']} nonfinite)")
        for x in s["budget_sensitivity"]:
            print(f"    sensitivity {x['rule_version']:<16} floor {str(x['loss_floor']):<7} "
                  f"stable {x['stable_at_largest']}/{x['eligible_at_largest']}  "
                  f"sufficient={x['largest_candidate_sufficient']}  shortest={x['recommended_budget']}")
        c = s["calibration"]
        print(f"  thresholds changed: {', '.join(c['changed']) or 'none'}; "
              f"manual review: {', '.join(c['manual_review']) or 'none'}")
        for arm, v in s["outcomes_proposed"]["arms"].items():
            print(f"  {arm:<10} n={v['n']}  " + "  ".join(
                f"{o}={x['count']}" for o, x in v["outcomes"].items()) +
                f"  born_dead={v['born_dead']['count']}")
        print(f"  report: {exp.root / ANALYSIS_DIRNAME / 'pilot_report.md'}")
        return 0

    if args.cmd == "preview":
        try:
            pr = preview_freeze(args.proposed, args.target, args.expected_hash)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"PREVIEW (nothing written). Proposal {pr['proposal_hash'][:12]} -> {pr['target']} "
              f"(currently frozen: {pr['target_frozen']})")
        print(json.dumps({k: v for k, v in pr["would_write"].items() if k != "frozen_note"}, indent=2))
        print("changes vs current file: " + (", ".join(f"{k}: {a} -> {b}" for k, (a, b) in pr["changes"].items())
                                            or "none"))
        return 0

    try:
        th = freeze_thresholds(args.proposed, args.target, args.note, args.expected_hash)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"froze {args.target} (hash {th.hash()[:12]}): {th.frozen_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
