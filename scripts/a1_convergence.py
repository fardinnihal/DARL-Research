"""Convergence diagnostics for a finished A1 ReLU pilot (decides nothing).

    python scripts/a1_convergence.py --config configs/a1/pilot.json
    python scripts/a1_convergence.py --config configs/a1/pilot.json --budget 1500

Reads only validated runs (same checks as a1_pilot.py analyze), uses training
MSE only, and writes analysis/convergence_diagnostics.{json,md} (or
..._B<budget>.* with --budget). The approved budget rule is not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from darl_study.a1.checkpoint import atomic_write_text
from darl_study.a1.convergence import convergence_diagnostics, diagnostics_markdown
from darl_study.a1.outcomes import load_thresholds
from darl_study.a1.pilot import load_pilot_runs
from darl_study.a1.runner import load_experiment

REPO = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A1 pilot convergence diagnostics (training MSE only).")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "a1" / "pilot.json")
    ap.add_argument("--budget", type=int, default=None, help="epoch to diagnose (default: last)")
    args = ap.parse_args(argv)
    exp = load_experiment(args.config)
    th = load_thresholds(exp.path(exp.thresholds), purpose=exp.purpose)
    try:
        runs, _ = load_pilot_runs(exp, th)
        d = convergence_diagnostics(runs, th, budget=args.budget)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    tag = "" if args.budget is None else f"_B{args.budget}"
    out = exp.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / f"convergence_diagnostics{tag}.json", json.dumps(d, indent=2))
    md = diagnostics_markdown(d, f"A1 convergence diagnostics: {exp.name}"
                                 f"{'' if args.budget is None else f' at epoch {args.budget}'}")
    atomic_write_text(out / f"convergence_diagnostics{tag}.md", md)
    print(md)
    print(f"wrote {out / f'convergence_diagnostics{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
