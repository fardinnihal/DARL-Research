"""Run an A1 experiment from a JSON config (resumable; completed runs are skipped).

Usage (from the repository root):
    python scripts/a1_run.py --config configs/a1/smoke.json
    python scripts/a1_run.py --config configs/a1/smoke.json --workers 3
    python scripts/a1_run.py --config configs/a1/smoke.json --arms ReLU-He DARL-He
    python scripts/a1_run.py --config configs/a1/smoke.json --dry-run

Re-running the same command resumes interrupted runs from their last
checkpoint and skips finished ones whose config and threshold hashes match.
Press Ctrl+C to stop; nothing finished is lost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from darl_study.a1.runner import load_experiment, run_experiment


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run an A1 experiment.")
    ap.add_argument("--config", type=Path, required=True, help="experiment JSON, e.g. configs/a1/smoke.json")
    ap.add_argument("--arms", nargs="+", default=None, help="run only these arm names")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per process (default 1)")
    ap.add_argument("--dry-run", action="store_true", help="list runs and their status; train nothing")
    args = ap.parse_args(argv)

    exp = load_experiment(args.config)
    try:
        results = run_experiment(exp, arm_names=args.arms, workers=args.workers,
                                 threads=args.threads, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\ninterrupted: rerun the same command to resume from the last checkpoints", file=sys.stderr)
        return 130
    return 1 if any(r["action"] == "conflict" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
