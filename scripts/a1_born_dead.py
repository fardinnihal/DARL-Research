"""A1 initialization-only born-dead experiment (no training).

For each run seed and init scheme, builds the 10-layer width-2 network,
initialises it, and checks on the fixed training inputs (Lu eq. 3.1 is defined
on the training inputs):

* dead_layer     EXACT structural test: some hidden layer has z <= 0 for every
                 unit on every training input. For ReLU this makes the network
                 output exactly constant (born dead).
* first_dead     index (1-based) of the first such layer, or 0.
* out_range      max - min of the network output over the training inputs.
* const_output   NUMERICAL PROXY: out_range < 1e-6.
* dead_units     number of hidden units (of 18) with z <= 0 on all inputs.

Lu et al. report that 5.6% (He) and 18.3% (RAI) of runs collapsed after
training without being born dead, out of 93.6% and 40.3% collapsed. That
implies born-dead rates of 88.0% (He) and 22.0% (RAI). These are reference
values only; nothing here is tuned to match them.

Usage (from the repository root):
    python scripts/a1_born_dead.py --seeds 20                 # quick check
    python scripts/a1_born_dead.py --seeds 1000               # full check
    python scripts/a1_born_dead.py --seeds 1000 --activations relu leaky_relu darl

Outputs go to results/a1/born_dead/<tag>/ (git-ignored):
    per_seed.csv   one row per (activation, init, seed)
    summary.csv    counts, rates and 95% Clopper-Pearson intervals
    summary.json   the same plus config, dataset hashes and versions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from darl_study import __version__ as PKG_VERSION
from darl_study.a1.data import build_a1_datasets, load_data_config
from darl_study.a1.init import INIT_SCHEMES, derive_seeds, initialize_
from darl_study.a1.model import ACTIVATIONS, DeepNarrowMLP

REPO = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO / "configs" / "a1" / "data.json"
CONST_TOL = 1e-6  # numerical proxy threshold for "constant output"

# Implied by Lu et al. Fig. 6 and Section 5 text (collapsed minus not-born-dead).
LU_IMPLIED_BORN_DEAD = {"he": 0.880, "rai": 0.220}

DEFAULT_ACT_KWARGS = {"relu": {}, "leaky_relu": {"negative_slope": 0.01}, "darl": {}}


# ------------------------------------------------------------- statistics
def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k + 1)))


def _bisect(f, target: float, increasing: bool) -> float:
    """Solve f(p) = target on [0, 1] for monotone f."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson(k: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Exact two-sided binomial confidence interval for k successes in n trials.

    lower: P(X >= k | p) = (1 - level) / 2   (increasing in p)
    upper: P(X <= k | p) = (1 - level) / 2   (decreasing in p)
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError("need n >= 1 and 0 <= k <= n")
    a = (1.0 - level) / 2.0
    lower = 0.0 if k == 0 else _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p), a, increasing=True)
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p), a, increasing=False)
    return lower, upper


# ---------------------------------------------------------------- checks
@torch.no_grad()
def born_dead_record(model: DeepNarrowMLP, x: torch.Tensor) -> Dict[str, float]:
    """Structural and numerical born-dead checks on inputs x, in eval mode."""
    was_training = model.training
    model.eval()  # DARL state is frozen in eval; nothing is updated here
    try:
        out, pre = model.forward_with_preactivations(x)
    finally:
        model.train(was_training)
    unit_dead = [(z <= 0).all(dim=0) for z in pre]  # per layer: (width,) bool
    layer_dead = [bool(d.all()) for d in unit_dead]
    first = next((i + 1 for i, d in enumerate(layer_dead) if d), 0)
    out_range = float(out.max() - out.min())
    if not math.isfinite(out_range):
        raise FloatingPointError("non-finite network output at initialization")
    return {
        "dead_layer": int(first > 0),
        "first_dead": first,
        "dead_units": int(sum(int(d.sum()) for d in unit_dead)),
        "out_range": out_range,
        "const_output": int(out_range < CONST_TOL),
    }


def run(
    seeds: Sequence[int],
    inits: Sequence[str],
    activations: Sequence[str],
    x: torch.Tensor,
    progress: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(inits) * len(activations)
    done, t0 = 0, time.perf_counter()
    for act in activations:
        for scheme in inits:
            for s in seeds:
                model = DeepNarrowMLP(activation=act, act_kwargs=DEFAULT_ACT_KWARGS[act])
                derived = initialize_(model, scheme, s)
                rec = born_dead_record(model, x)
                rows.append({"activation": act, "init": scheme, "seed": s,
                             "init_seed": derived["init"], **rec})
                done += 1
                if progress and (done % 200 == 0 or done == total):
                    rate = done / max(time.perf_counter() - t0, 1e-9)
                    print(f"  {done}/{total} networks checked ({rate:.0f}/s)", flush=True)
    return rows


def summarise(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    keys = sorted({(r["activation"], r["init"]) for r in rows},
                  key=lambda t: (ACTIVATIONS.index(t[0]), INIT_SCHEMES.index(t[1])))
    for act, scheme in keys:
        grp = [r for r in rows if r["activation"] == act and r["init"] == scheme]
        n = len(grp)
        row: Dict[str, object] = {"activation": act, "init": scheme, "n": n}
        for metric in ("dead_layer", "const_output"):
            k = sum(int(r[metric]) for r in grp)
            lo, hi = clopper_pearson(k, n)
            row.update({f"{metric}_count": k, f"{metric}_rate": k / n,
                        f"{metric}_ci95_low": lo, f"{metric}_ci95_high": hi})
        row["dead_and_const_disagree"] = sum(int(r["dead_layer"]) != int(r["const_output"]) for r in grp)
        row["mean_dead_units"] = sum(int(r["dead_units"]) for r in grp) / n
        ref = LU_IMPLIED_BORN_DEAD.get(scheme) if act == "relu" else None
        row["lu_implied_born_dead"] = ref
        row["lu_ref_inside_ci95"] = (None if ref is None else
                                     bool(row["dead_layer_ci95_low"] <= ref <= row["dead_layer_ci95_high"]))
        out.append(row)
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=1000, help="number of run seeds (default 1000)")
    ap.add_argument("--start", type=int, default=0, help="first run seed (default 0)")
    ap.add_argument("--inits", nargs="+", default=list(INIT_SCHEMES), choices=INIT_SCHEMES)
    ap.add_argument("--activations", nargs="+", default=["relu"], choices=ACTIVATIONS)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default results/a1/born_dead/seeds<start>-<end>)")
    args = ap.parse_args(argv)
    if args.seeds < 1 or args.start < 0:
        ap.error("--seeds must be >= 1 and --start >= 0")

    torch.set_num_threads(1)
    seeds = list(range(args.start, args.start + args.seeds))
    out_dir = args.out or REPO / "results" / "a1" / "born_dead" / f"seeds{seeds[0]}-{seeds[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_data_config(DATA_CONFIG)
    train = build_a1_datasets(cfg)["train"]
    print(f"A1 born-dead check: {len(seeds)} seeds x inits {args.inits} x activations {args.activations}")
    print(f"training inputs: n={len(train)}, sha256={train.fingerprint()[:16]}...")

    t0 = time.perf_counter()
    rows = run(seeds, args.inits, args.activations, train.x)
    elapsed = time.perf_counter() - t0
    summary = summarise(rows)

    _write_csv(out_dir / "per_seed.csv", rows)
    _write_csv(out_dir / "summary.csv", summary)
    meta = {
        "experiment": "A1 initialization-only born-dead check",
        "seeds": [seeds[0], seeds[-1]],
        "inits": args.inits,
        "activations": args.activations,
        "act_kwargs": {a: DEFAULT_ACT_KWARGS[a] for a in args.activations},
        "network": DeepNarrowMLP().describe(),
        "evaluated_on": "training inputs (Lu eq. 3.1)",
        "const_output_tol": CONST_TOL,
        "data_config": cfg.to_dict(),
        "train_sha256": train.fingerprint(),
        "example_derived_seeds": {str(seeds[0]): derive_seeds(seeds[0])},
        "versions": {"darl_study": PKG_VERSION, "torch": torch.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0],
                     "platform": platform.platform()},
        "elapsed_seconds": round(elapsed, 3),
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print(f"{'activation':<11} {'init':<4} {'n':>5} | {'dead layer':>10} {'95% CI':>15} | "
          f"{'const out':>9} | {'Lu implied':>10}")
    for r in summary:
        ref = r["lu_implied_born_dead"]
        ref_s = "-" if ref is None else f"{100 * ref:.1f}%"
        print(f"{r['activation']:<11} {r['init']:<4} {r['n']:>5} | "
              f"{100 * r['dead_layer_rate']:>9.1f}% "
              f"({100 * r['dead_layer_ci95_low']:>5.1f}-{100 * r['dead_layer_ci95_high']:>5.1f}) | "
              f"{100 * r['const_output_rate']:>8.1f}% | {ref_s:>10}")
    print(f"\nwrote {out_dir}  ({elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
