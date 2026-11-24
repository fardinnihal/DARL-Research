"""Measure A1 training cost on this CPU and project run-mode budgets.

Runs a short training (default 20 epochs, 2 seeds, after one untimed
warm-up) for each requested arm through the real training loop, with the
pilot's checkpoint cadence and the runner's A1 metrics hook. It times one measurement record separately, so
the cost splits into training per epoch plus measurement per record, and a
2,000-epoch run is projected with its real number of records (204). It then projects wall-clock time for
the planned run modes. Nothing here changes any scientific setting; timing
runs use temporary folders that are deleted afterwards.

Usage (from the repository root):
    python scripts/a1_time.py
    python scripts/a1_time.py --epochs 50 --seeds 3 --workers 6

Writes results/a1/timing/timing.json (git-ignored).

Projections assume one single-threaded process per worker and linear scaling
across workers. Real scaling depends on your CPU (physical cores, turbo,
thermals); the pilot will confirm it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from darl_study.a1.data import build_a1_datasets
from darl_study.a1.init import INIT_SCHEMES, initialize_
from darl_study.a1.metrics import a1_extra_metrics
from darl_study.a1.model import ACTIVATIONS, DeepNarrowMLP
from darl_study.a1.train import TrainConfig, evaluate, log_epochs, train

REPO = Path(__file__).resolve().parents[1]
ACT_KWARGS = {"relu": {}, "leaky_relu": {"negative_slope": 0.01}, "darl": {}}


def _eval_seconds(act: str, init: str, reps: int = 20) -> float:
    """Cost of one measurement record as the runner makes it: full train/val/probe
    passes, gradient, and the A1 neuron and shape metrics (``a1_extra_metrics``)."""
    model = DeepNarrowMLP(activation=act, act_kwargs=ACT_KWARGS[act])
    initialize_(model, init, 0)
    data = build_a1_datasets()
    evaluate(model, data, 0, 0, extra_metrics=a1_extra_metrics)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        evaluate(model, data, 0, 0, extra_metrics=a1_extra_metrics)
    return (time.perf_counter() - t0) / reps


def measure(act: str, init: str, seeds: Sequence[int], epochs: int,
            log_every: int, checkpoint_every: int, budget_epochs: int) -> Dict[str, float]:
    """Time real training runs and split the cost into per-epoch training and
    per-record measurement, so short timing runs project correctly to long ones."""
    eval_s = _eval_seconds(act, init)
    n_records = len(log_epochs(epochs, log_every))
    per_epoch: List[float] = []
    steps = 0
    with tempfile.TemporaryDirectory() as tmp:
        warm = TrainConfig(activation=act, act_kwargs=ACT_KWARGS[act], init=init, run_seed=999_999, epochs=2)
        train(warm, extra_metrics=a1_extra_metrics)  # warm-up, not timed
        for s in seeds:
            cfg = TrainConfig(activation=act, act_kwargs=ACT_KWARGS[act], init=init, run_seed=s,
                              epochs=epochs, log_every=log_every, checkpoint_every=checkpoint_every)
            t0 = time.perf_counter()
            r = train(cfg, run_dir=Path(tmp) / f"seed{s}", extra_metrics=a1_extra_metrics)
            dt = time.perf_counter() - t0
            done = max(r["epochs_completed"], 1)
            per_epoch.append(max(dt - n_records * eval_s, 0.0) / done)
            steps = r["steps_completed"] // done
    train_epoch = sum(per_epoch) / len(per_epoch)
    run_s = budget_epochs * train_epoch + len(log_epochs(budget_epochs, log_every)) * eval_s
    return {"activation": act, "init": init,
            "train_seconds_per_epoch": train_epoch,
            "ms_per_step": 1000.0 * train_epoch / max(steps, 1),
            "steps_per_epoch": steps,
            "eval_seconds_per_record": eval_s,
            "projected_seconds_per_run": run_s,
            "train_seconds_per_epoch_each_seed": per_epoch}


def fmt_hours(sec: float) -> str:
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Time A1 training on this CPU.")
    ap.add_argument("--epochs", type=int, default=20, help="epochs per timing run (default 20)")
    ap.add_argument("--seeds", type=int, default=2, help="timing runs per arm (default 2)")
    ap.add_argument("--activations", nargs="+", default=list(ACTIVATIONS), choices=ACTIVATIONS)
    ap.add_argument("--inits", nargs="+", default=list(INIT_SCHEMES), choices=INIT_SCHEMES)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per process (default 1)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes assumed in projections (default: CPUs - 1)")
    ap.add_argument("--budget-epochs", type=int, default=2000,
                    help="epochs per run in projections (default 2000 = pilot budget, an upper bound for main)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "a1" / "timing" / "timing.json")
    args = ap.parse_args(argv)
    if min(args.epochs, args.seeds, args.threads, args.workers, args.budget_epochs) < 1:
        ap.error("all counts must be >= 1")

    torch.set_num_threads(args.threads)
    print(f"A1 timing: {args.epochs} epochs x {args.seeds} seeds per arm, "
          f"torch threads={args.threads}, CPUs reported={os.cpu_count()}")
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, {platform.platform()}\n")

    rows = []
    for act in args.activations:
        for init in args.inits:
            r = measure(act, init, range(1000, 1000 + args.seeds), args.epochs,
                        args.log_every, args.checkpoint_every, args.budget_epochs)
            rows.append(r)
            print(f"  {act:<11} {init:<4} {r['ms_per_step']:5.2f} ms/step  "
                  f"{r['train_seconds_per_epoch'] * 1000:6.1f} ms/epoch  "
                  f"{r['eval_seconds_per_record'] * 1000:6.1f} ms/record  "
                  f"-> {r['projected_seconds_per_run']:6.1f} s per {args.budget_epochs}-epoch run", flush=True)

    def run_seconds(act: str, init: str) -> float:
        m = [r for r in rows if r["activation"] == act and r["init"] == init]
        return m[0]["projected_seconds_per_run"] if m else float("nan")

    plans = []
    def plan(name: str, arms: List[tuple], seeds: int) -> None:
        cpu = sum(run_seconds(a, i) for a, i in arms) * seeds
        if cpu == cpu:  # skip plans whose arms were not timed
            plans.append({"plan": name, "arms": [f"{a}-{i}" for a, i in arms], "seeds_per_arm": seeds,
                          "epochs": args.budget_epochs, "cpu_seconds": cpu,
                          "wall_seconds": cpu / args.workers, "workers": args.workers})

    relu_ref = [("relu", "he"), ("relu", "rai")]
    plan("pilot: ReLU He + RAI", relu_ref, 20)
    plan("Step 9 calibration: DARL He + RAI", [("darl", "he"), ("darl", "rai")], 20)
    for init in ("he", "rai"):
        comp = [("relu", init), ("leaky_relu", init), ("darl", init), ("leaky_relu", init)]  # last = slope-matched
        plan(f"Step 9 main, {init} set: ReLU, LReLU 0.01, DARL, LReLU matched", comp, 1000)
    plan("Step 9 main, all 8 arms", [(a, i) for i in ("he", "rai")
                                     for a in ("relu", "leaky_relu", "darl", "leaky_relu")], 1000)
    darl = [r for r in rows if r["activation"] == "darl"]
    for r in darl:
        base = [b for b in rows if b["activation"] == "relu" and b["init"] == r["init"]]
        if base:
            print(f"  DARL overhead vs ReLU ({r['init']}): training step x"
                  f"{r['ms_per_step'] / base[0]['ms_per_step']:.2f}, measurement record x"
                  f"{r['eval_seconds_per_record'] / base[0]['eval_seconds_per_record']:.2f}")

    print(f"\nProjected time at {args.budget_epochs} epochs/run, {args.workers} workers "
          "(single-thread processes, linear scaling assumed):")
    for p in plans:
        print(f"  {p['plan']:<70} {p['seeds_per_arm']:>5} seeds/arm  "
              f"CPU {fmt_hours(p['cpu_seconds']):>9}  wall ~{fmt_hours(p['wall_seconds']):>9}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(),
                    "processor": platform.processor(), "python": sys.version.split()[0],
                    "torch": torch.__version__},
        "measurements": rows,
        "projections": plans,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
