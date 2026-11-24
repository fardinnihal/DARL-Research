"""Convergence DIAGNOSTICS for A1 pilot runs: still improving, plateaued or noisy?

This module decides nothing. The budget decision is made only by
``darl_study.a1.pilot.budget_check`` (rule v2: final training MSE <= 1e-5 or
training-MSE change < 1% over the last 10% of epochs, for >= 95% of
non-collapsed runs). These diagnostics explain how runs behave against the
1% change clause (``approved_change``/``approved_pass`` below; this clause
alone was rule v1), using only full-batch training MSE (``train_mse``),
never validation data.

For each finished, non-collapsed run and a budget B, over the window
[w0, B] with w0 = B - ``trend_frac`` * B (default: the last 20% of epochs,
every recorded epoch in it, i.e. 41 points at B = 2,000 with log_every 10):

  approved_change   |L(B) - L(0.9 B)| / L(0.9 B)   (the approved rule, for reference)
  slope             least-squares slope of log10 L(e) on e, in decades per
                    1,000 epochs (negative = loss still falling)
  slope_t           slope / its standard error
  noise_pct         typical relative scatter of one record about the trend:
                    100 * (10 ** resid_sd - 1)
  pass_rate_if_flat probability that two independent points with this
                    scatter differ by less than 1% (normal model): how often a
                    perfectly flat run with this noise would pass the rule
  best_gain_pct     how much the best (lowest) loss seen improved in the last
                    10% of epochs: 100 * (1 - min L[0.9B, B] / min L[0, 0.9B))
  projected_pct     trend-projected relative change over the NEXT 10% of B

  status            "still_improving" : slope_t <= -t_crit and best_gain_pct >= 1
                    "rising"          : slope_t >=  t_crit
                    "plateaued"       : otherwise (no significant trend)

These are statistical descriptions of the recorded curve, not definitions
from Lu et al.; they are diagnostics for the reviewer only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from darl_study.a1.outcomes import OutcomeThresholds
from darl_study.a1.pilot import _collapsed, plateau_change, record_at

__all__ = ["run_diagnostics", "convergence_diagnostics", "diagnostics_markdown"]


def _pass_rate_if_flat(resid_sd_dec: float, tol: float) -> float:
    """P(|log ratio of two points| < log10(1 + tol)) for i.i.d. normal log10 scatter."""
    if resid_sd_dec <= 0:
        return 1.0
    sd_diff = math.sqrt(2.0) * resid_sd_dec
    z = math.log10(1.0 + tol) / sd_diff
    return math.erf(z / math.sqrt(2.0))


def run_diagnostics(history: Sequence[Dict[str, Any]], budget: int, tol: float = 0.01,
                    window_frac: float = 0.1, trend_frac: float = 0.2, t_crit: float = 3.0) -> Dict[str, Any]:
    w0 = budget - int(round(trend_frac * budget))
    start = budget - int(round(window_frac * budget))
    pts = [(r["epoch"], float(r["train_mse"])) for r in history if w0 <= r["epoch"] <= budget]
    if len(pts) < 5 or record_at(history, budget) is None:
        raise ValueError(f"too few records in [{w0}, {budget}]")
    if any(v <= 0 or not math.isfinite(v) for _, v in pts):
        raise ValueError("training MSE must be positive and finite for a log-scale trend")
    e = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.log10(np.array([p[1] for p in pts], dtype=np.float64))
    ec = e - e.mean()
    sxx = float((ec ** 2).sum())
    slope = float((ec * (y - y.mean())).sum() / sxx)              # decades per epoch
    resid = y - (y.mean() + slope * ec)
    dof = len(pts) - 2
    resid_sd = float(math.sqrt((resid ** 2).sum() / dof))
    se = resid_sd / math.sqrt(sxx) if resid_sd > 0 else 0.0
    slope_t = slope / se if se > 0 else (-math.inf if slope < 0 else (math.inf if slope > 0 else 0.0))
    before = [float(r["train_mse"]) for r in history if r["epoch"] < start]
    last = [float(r["train_mse"]) for r in history if start <= r["epoch"] <= budget]
    best_gain = 100.0 * (1.0 - min(last) / min(before)) if before and min(before) > 0 else None
    projected = 100.0 * (10 ** (slope * window_frac * budget) - 1.0)
    if slope_t <= -t_crit and best_gain is not None and best_gain >= 100.0 * tol:
        status = "still_improving"
    elif slope_t >= t_crit:
        status = "rising"
    else:
        status = "plateaued"
    return {
        "budget": budget,
        "final_train_mse": float(record_at(history, budget)["train_mse"]),
        "approved_change": plateau_change(history, budget, window_frac),
        "approved_pass": plateau_change(history, budget, window_frac) < tol,
        "slope_dec_per_1000": 1000.0 * slope,
        "slope_t": slope_t,
        "noise_pct": 100.0 * (10 ** resid_sd - 1.0),
        "pass_rate_if_flat": _pass_rate_if_flat(resid_sd, tol),
        "best_gain_pct": best_gain,
        "projected_pct_next_window": projected,
        "status": status,
        "n_points": len(pts),
    }


def convergence_diagnostics(runs: Sequence[Dict[str, Any]], th: OutcomeThresholds,
                            budget: Optional[int] = None, **kw: Any) -> Dict[str, Any]:
    """Diagnostics for every completed, non-collapsed run at ``budget`` (default: its last epoch)."""
    rows: List[Dict[str, Any]] = []
    skipped = {"collapsed": 0, "nonfinite": 0}
    for r in runs:
        if r["status"] != "completed":
            skipped["nonfinite"] += 1
            continue
        b = budget if budget is not None else r["epochs"]
        rec = record_at(r["history"], b)
        if rec is None:
            raise ValueError(f"{r['arm']} seed {r['seed']}: no record at epoch {b}")
        if _collapsed(rec, th):
            skipped["collapsed"] += 1
            continue
        rows.append({"arm": r["arm"], "seed": r["seed"], **run_diagnostics(r["history"], b, **kw)})
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    fails = [r for r in rows if not r["approved_pass"]]
    return {
        "note": "DIAGNOSTIC ONLY - the approved budget rule is unchanged and decides the budget",
        "budget": budget, "n_non_collapsed": len(rows), "skipped": skipped, "status_counts": counts,
        "approved_pass": sum(r["approved_pass"] for r in rows),
        "approved_fail_by_status": {s: sum(1 for r in fails if r["status"] == s)
                                    for s in ("still_improving", "plateaued", "rising")},
        "expected_pass_if_all_flat": sum(r["pass_rate_if_flat"] for r in rows),
        "rows": sorted(rows, key=lambda x: (x["arm"], x["seed"])),
    }


def _f(v: Any, fmt: str) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return format(v, fmt)


def diagnostics_markdown(d: Dict[str, Any], title: str) -> str:
    L = [f"# {title}", "", "**DIAGNOSTIC ONLY.** The approved budget rule is unchanged; this explains its result.",
         "Training MSE only (full batch); no validation data.", "",
         f"Non-collapsed runs: {d['n_non_collapsed']} (skipped: {d['skipped']['collapsed']} collapsed, "
         f"{d['skipped']['nonfinite']} nonfinite). Approved rule passed: {d['approved_pass']}.",
         f"Status: {', '.join(f'{k} {v}' for k, v in sorted(d['status_counts'].items())) or 'none'}.",
         f"Failures of the approved rule by status: {d['approved_fail_by_status']}.",
         f"Expected passes if every run were perfectly flat with its measured noise: "
         f"{d['expected_pass_if_all_flat']:.1f} of {d['n_non_collapsed']}.", "",
         "| Arm | Seed | Final train MSE | Approved change | Pass | Slope (dec/1000 ep) | t | Noise % | "
         "P(pass if flat) | Best-loss gain % (last 10%) | Projected % next 10% | Status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        L.append(f"| {r['arm']} | {r['seed']} | {r['final_train_mse']:.3g} | {_f(r['approved_change'], '.3g')} | "
                 f"{'yes' if r['approved_pass'] else 'no'} | {r['slope_dec_per_1000']:+.3f} | {_f(r['slope_t'], '.1f')} | "
                 f"{r['noise_pct']:.2f} | {r['pass_rate_if_flat']:.2f} | {_f(r['best_gain_pct'], '.1f')} | "
                 f"{r['projected_pct_next_window']:+.1f} | {r['status']} |")
    return "\n".join(L) + "\n"
