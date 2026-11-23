"""ReLU pilot analysis for A1 (Stage 2, Step 7): budget check, threshold calibration, summary.

Reads finished pilot runs written by ``darl_study.a1.runner`` (ReLU arms only)
and produces:

* per-seed and per-checkpoint metric tables (training loss, validation loss,
  output spread and variance on the probe grid, outcome label);
* the approved budget check (below);
* proposed outcome thresholds, calibrated on ReLU pilot runs only;
* outcome counts per arm with exact 95% intervals.

It never trains, never touches run folders, and writes only to
``<experiment root>/analysis/``. Only experiments of mode "pilot" or
"calibration" are accepted, so pilot numbers cannot end up in research tables.

Budget criterion (approved roadmap, decision 1)
-----------------------------------------------
The main budget is the shortest candidate B where at least 95% of the
non-collapsed pilot runs have a training-loss change under 1% over the last
10% of epochs. Implemented as:

  L(e)      full-batch training MSE at epoch e (``train_mse``: eval mode, all
            3,000 training points). Only training loss is used.
  change(B) |L(B) - L(0.9 B)| / L(0.9 B)
  eligible  status "completed" and NOT collapsed at epoch B (collapse proxy
            with the given thresholds). Nonfinite runs are counted, not used.
  pass(B)   converged / eligible >= 0.95, with converged = change(B) < 0.01.

Candidates are the epochs B where both B and 0.9 B were recorded. With
constant-learning-rate Adam, the first B epochs of a 2,000-epoch run are
exactly a B-epoch run, so shorter budgets can be read from the same runs.
Recommended budget: the shortest B that passes AND every longer candidate
also passes (a project rule that ignores an isolated early pass). If the
largest candidate (2,000) fails, 2,000 epochs are reported as insufficient.

Threshold calibration (proposal only; freezing is a separate, explicit step)
--------------------------------------------------------------------------
success_val_mse is NOT calibrated. It keeps its predetermined value (1e-3).
It is a validation-set threshold and the validation set is for reporting only
(data.py), so no pilot distribution moves it; and training MSE must not stand
in for it, because the classifier compares it with validation MSE. Training
MSE is used only by the budget check above.

The other thresholds cut probe-grid or training-input quantities. Each is
checked against the pilot distribution of the quantity it cuts, on log10
scale, in this order (later steps use the earlier proposals):

  collapse_std_frac   out_std / target_std on the probe grid, final epoch,
                      all completed runs.
  half_fit_mse        MSE of the better-fitted half of the probe grid,
                      all non-collapsed runs.
  half_flat_std_frac  output std / target std on the other half, same runs.
  const_range_tol     network output range on the training inputs at init.

Validation MSE is never read by the calibration. The half-trained quantities
use all non-collapsed runs instead of only the non-success ones, because
picking the non-success runs would need validation MSE. Successful runs lie on
the fitted side of half_fit_mse and far above half_flat_std_frac, so they do
not blur either cut.

Rule: keep the draft value if no pilot value lies within ``min_margin``
decades (default 0.5) on either side. Otherwise propose the geometric
midpoint of the widest gap between consecutive pilot values, if that gap is
at least ``min_gap`` decades (default 1). Otherwise keep the draft and flag
it for manual review. Exact zeros are counted separately (they lie below any
positive threshold) and are left out of the gap search. Fewer than
``min_values`` (default 5) values also flag manual review.

Only runs whose result.json, outcome.json and DONE are all valid and agree
with the current training-config hash, thresholds hash and experiment record
are analysed (``load_pilot_runs``); anything else is listed and, unless
``allow_incomplete``, stops the analysis.

Every label here is a NUMERICAL PROXY on the probe grid or training set; see
``PROXY_VS_PAPER``.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from darl_study.a1.checkpoint import atomic_write_text, config_hash
from darl_study.a1.data import load_data_config
from darl_study.a1.outcomes import OUTCOMES, OutcomeThresholds, classify_shape, write_thresholds
from darl_study.a1.runner import (
    DONE_NAME,
    OUTCOME_NAME,
    ExperimentConfig,
    _read_json,
    _valid_outcome,
    _valid_result,
    run_dir_for,
)
from darl_study.a1.train import RESULT_NAME

__all__ = [
    "PILOT_MODES",
    "PROXY_VS_PAPER",
    "clopper_pearson",
    "experiment_hash",
    "load_pilot_runs",
    "record_at",
    "seed_table",
    "checkpoint_table",
    "plateau_change",
    "candidate_budgets",
    "budget_check",
    "gap_report",
    "calibrate_thresholds",
    "outcome_summary",
    "analyze_pilot",
    "freeze_thresholds",
]

PILOT_MODES = ("pilot", "calibration")
PILOT_ACTIVATIONS = ("relu",)
DEFAULT_CHECKPOINTS = (0, 10, 100, 250, 500, 1000, 1500, 2000)
ZERO = 0.0

PROXY_VS_PAPER = [
    ("Collapse (Lu Fig. 6 A)", "the trained network is a constant function",
     "out_std < collapse_std_frac * target_std on the 2,001-point probe grid"),
    ("Success (Lu Fig. 6 C)", "the network fits |x| (judged visually)",
     "validation MSE < success_val_mse"),
    ("Half-trained (Lu Fig. 6 B)", "one side of the V fitted, the other flat (visual)",
     "one probe half MSE < half_fit_mse and the other half's output std < half_flat_std_frac * its target std"),
    ("Other / nonfinite", "not in Lu",
     "finite but none of the above / the run stopped on NaN or Inf"),
    ("Born dead (Lu eq. 3.1)", "network constant on the training inputs at init",
     "ReLU: some hidden layer has z <= 0 for all units on all 3,000 training inputs (exact for ReLU)"),
    ("Dead unit", "unit inactive for all inputs",
     "preactivation z <= 0 at every probe point (q == 1)"),
    ("Converged (budget)", "not defined by Lu (training length not stated)",
     "full-batch training MSE changes < 1% over the last 10% of epochs"),
]


# --------------------------------------------------------------- statistics
def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * lp + (n - i) * lq) for i in range(k + 1)))


def _bisect(f, target: float, increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson(k: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Exact two-sided binomial interval (same definition as scripts/a1_born_dead.py)."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError("need n >= 1 and 0 <= k <= n")
    a = (1.0 - level) / 2.0
    lower = 0.0 if k == 0 else _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p), a, increasing=True)
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p), a, increasing=False)
    return lower, upper


# ------------------------------------------------------------------ loading
def experiment_hash(exp: ExperimentConfig) -> str:
    """Same hash the runner writes to experiment.json (seed_count excluded)."""
    return config_hash({k: v for k, v in exp.to_dict().items() if k != "seed_count"})


def load_pilot_runs(exp: ExperimentConfig, th: OutcomeThresholds,
                    allow_incomplete: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Finished pilot runs of every (arm, seed), plus a list of problems.

    ``th`` are the thresholds the runs were classified with (the experiment's
    thresholds file). The experiment record must exist and match this
    experiment and ``th``. A run counts only if result.json is a valid terminal
    result of this exact training config AND outcome.json and DONE are valid
    and consistent with it, with the config hash and with ``th`` (the same
    checks the runner uses before it skips a run), and outcome.json names this
    arm and seed. Raises if any run is unusable, unless ``allow_incomplete``.
    """
    if exp.mode not in PILOT_MODES:
        raise ValueError(f"pilot analysis accepts modes {PILOT_MODES} only, got {exp.mode!r}; "
                         "pilot numbers must stay separate from research results")
    bad = [a.name for a in exp.arms if a.activation not in PILOT_ACTIVATIONS]
    if bad:
        raise ValueError(f"the threshold pilot uses ReLU arms only; non-ReLU arms: {bad}")
    record = _read_json(exp.root / "experiment.json")
    if record is None:
        raise ValueError(f"{exp.root / 'experiment.json'} is missing or invalid; run a1_run.py first")
    if record.get("experiment_hash") != experiment_hash(exp):
        raise ValueError(f"{exp.root} holds a different experiment (experiment hash mismatch)")
    if record.get("thresholds_hash") != th.hash():
        raise ValueError("the thresholds differ from the ones this pilot was run with "
                         "(experiment.json thresholds_hash mismatch); rerun a1_run.py to re-classify")
    thash = th.hash()
    data_cfg = load_data_config(exp.path(exp.data_config))
    runs, problems = [], []
    for arm in exp.arms:
        for seed in exp.seeds:
            d = run_dir_for(exp, arm, seed)
            chash = exp.train_config(arm, seed, data_cfg).hash()
            res = _read_json(d / RESULT_NAME)
            out = _read_json(d / OUTCOME_NAME)
            done = _read_json(d / DONE_NAME)
            if not _valid_result(res, chash):
                problems.append(f"{arm.name} seed {seed}: no valid finished result.json for this config")
                continue
            if not _valid_outcome(out, done, res, chash, thash):
                problems.append(f"{arm.name} seed {seed}: outcome.json/DONE missing, corrupt or inconsistent "
                                "with the config/thresholds hashes (rerun a1_run.py to regenerate)")
                continue
            if out.get("arm") != arm.name or out.get("seed") != seed:
                problems.append(f"{arm.name} seed {seed}: outcome.json names a different arm or seed")
                continue
            born = out.get("born_dead_record")
            if born is not None and not isinstance(born, dict):
                problems.append(f"{arm.name} seed {seed}: outcome.json born_dead_record is malformed")
                continue
            runs.append({"arm": arm.name, "activation": arm.activation, "init": arm.init, "seed": seed,
                         "status": res["status"], "history": res["history"], "failure": res.get("failure"),
                         "epochs": res["config"]["epochs"], "config_hash": chash,
                         "born": out.get("born_dead_record")})
    if problems and not allow_incomplete:
        raise ValueError(f"{len(problems)} pilot run(s) unusable:\n  " + "\n  ".join(problems[:20]))
    return runs, problems


def record_at(history: Sequence[Dict[str, Any]], epoch: int) -> Optional[Dict[str, Any]]:
    for rec in history:
        if rec["epoch"] == epoch:
            return rec
    return None


def _label(rec: Optional[Dict[str, Any]], th: OutcomeThresholds) -> Optional[str]:
    if rec is None or "shape" not in rec:
        return None
    return classify_shape(rec["shape"], rec["val_mse"], th)


def _final_label(run: Dict[str, Any], th: OutcomeThresholds) -> str:
    if run["status"] == "nonfinite":
        return "nonfinite"
    return _label(run["history"][-1], th)


def _collapsed(rec: Dict[str, Any], th: OutcomeThresholds) -> bool:
    s = rec["shape"]
    return s["out_std"] < th.collapse_std_frac * s["target_std"]


# ------------------------------------------------------------------ tables
def seed_table(runs: Sequence[Dict[str, Any]], th: OutcomeThresholds) -> List[Dict[str, Any]]:
    rows = []
    for r in runs:
        h = r["history"]
        f = h[-1] if h else {}
        s = f.get("shape", {})
        flags = [_collapsed(x, th) for x in h if "shape" in x]
        born = r["born"] or {}
        rows.append({
            "arm": r["arm"], "init": r["init"], "seed": r["seed"], "status": r["status"],
            "outcome": _final_label(r, th),
            "born_dead": born.get("born_dead"),
            "first_dead_layer_at_init": born.get("first_dead_layer"),
            "init_out_range": born.get("out_range"),
            "collapsed_at_init": flags[0] if flags else None,
            "first_collapse_epoch": next((x["epoch"] for x in h if "shape" in x and _collapsed(x, th)), None),
            "epochs_completed": f.get("epoch"),
            "final_train_mse": f.get("train_mse"),
            "final_train_loss_epoch": f.get("train_loss_epoch"),
            "final_val_mse": f.get("val_mse"),
            "final_probe_mse": s.get("probe_mse"),
            "final_out_std": s.get("out_std"),
            "final_out_var": None if s.get("out_std") is None else s["out_std"] ** 2,
            "final_left_mse": s.get("left_mse"), "final_right_mse": s.get("right_mse"),
            "final_left_out_std": s.get("left_out_std"), "final_right_out_std": s.get("right_out_std"),
            "dead_units_init": h[0]["neurons"]["dead_count"] if h and "neurons" in h[0] else None,
            "dead_units_final": f["neurons"]["dead_count"] if "neurons" in f else None,
            "failure_phase": (r["failure"] or {}).get("phase"),
        })
    return rows


def checkpoint_table(runs: Sequence[Dict[str, Any]], th: OutcomeThresholds,
                     epochs: Sequence[int] = DEFAULT_CHECKPOINTS) -> List[Dict[str, Any]]:
    """Long table: one row per (run, checkpoint epoch) that was recorded."""
    rows = []
    for r in runs:
        for e in epochs:
            rec = record_at(r["history"], e)
            if rec is None:
                continue
            s = rec.get("shape", {})
            rows.append({"arm": r["arm"], "seed": r["seed"], "epoch": e,
                         "train_mse": rec["train_mse"], "train_loss_epoch": rec.get("train_loss_epoch"),
                         "val_mse": rec["val_mse"], "out_std": s.get("out_std"),
                         "out_var": None if s.get("out_std") is None else s["out_std"] ** 2,
                         "dead_units": rec["neurons"]["dead_count"] if "neurons" in rec else None,
                         "outcome": _label(rec, th)})
    return rows


# ------------------------------------------------------------------ budget
def plateau_change(history: Sequence[Dict[str, Any]], budget: int, window_frac: float = 0.1,
                   key: str = "train_mse") -> Optional[float]:
    """|L(B) - L(B - window)| / L(B - window), or None if either epoch was not recorded."""
    start = budget - int(round(window_frac * budget))
    a, b = record_at(history, start), record_at(history, budget)
    if a is None or b is None or start >= budget:
        return None
    la, lb = float(a[key]), float(b[key])
    if la == 0.0:
        return 0.0 if lb == 0.0 else math.inf
    return abs(lb - la) / la


def candidate_budgets(runs: Sequence[Dict[str, Any]], window_frac: float = 0.1, step: int = 100) -> List[int]:
    """Multiples of ``step`` where every completed run recorded both B and B - window."""
    done = [r for r in runs if r["status"] == "completed"]
    if not done:
        return []
    top = min(r["epochs"] for r in done)
    out = []
    for b in range(step, top + 1, step):
        if all(plateau_change(r["history"], b, window_frac) is not None for r in done):
            out.append(b)
    return out


def budget_check(runs: Sequence[Dict[str, Any]], th: OutcomeThresholds,
                 budgets: Optional[Sequence[int]] = None, tol: float = 0.01,
                 required: float = 0.95, window_frac: float = 0.1) -> Dict[str, Any]:
    budgets = list(budgets) if budgets is not None else candidate_budgets(runs, window_frac)
    if not budgets:
        raise ValueError("no candidate budget has records at both B and 0.9 B")
    rows = []
    for b in budgets:
        n_nonfinite = sum(r["status"] == "nonfinite" for r in runs)
        elig, conv, collapsed, changes, worst = 0, 0, 0, [], []
        per_arm: Dict[str, List[int]] = {}
        for r in runs:
            if r["status"] != "completed":
                continue
            rec = record_at(r["history"], b)
            if rec is None or "shape" not in rec:
                raise ValueError(f"{r['arm']} seed {r['seed']}: no shape record at epoch {b}")
            if _collapsed(rec, th):
                collapsed += 1
                continue
            c = plateau_change(r["history"], b, window_frac)
            elig += 1
            ok = c < tol
            conv += ok
            changes.append(c)
            per_arm.setdefault(r["arm"], [0, 0])
            per_arm[r["arm"]][0] += ok
            per_arm[r["arm"]][1] += 1
            if not ok:
                worst.append({"arm": r["arm"], "seed": r["seed"], "change": c, "train_mse": rec["train_mse"]})
        frac = conv / elig if elig else None
        worst.sort(key=lambda w: -w["change"])
        changes.sort()
        rows.append({"budget": b, "eligible": elig, "converged": conv, "fraction": frac,
                     "passes": frac is not None and frac >= required,
                     "excluded_collapsed": collapsed, "excluded_nonfinite": n_nonfinite,
                     "median_change": changes[len(changes) // 2] if changes else None,
                     "max_change": changes[-1] if changes else None,
                     "per_arm": {a: {"converged": v[0], "eligible": v[1]} for a, v in per_arm.items()},
                     "not_converged": worst})
    recommended = None
    for i, row in enumerate(rows):
        if all(x["passes"] for x in rows[i:]):
            recommended = row["budget"]
            break
    top = rows[-1]
    return {"criterion": {"loss": "train_mse (full-batch, eval mode)", "tol": tol, "required": required,
                          "window_frac": window_frac,
                          "rule": "shortest B that passes, with every longer candidate also passing"},
            "budgets": rows, "largest_candidate": top["budget"],
            "largest_candidate_sufficient": top["passes"], "recommended_budget": recommended}


# ------------------------------------------------------------- calibration
def gap_report(values: Sequence[float], current: float, min_margin: float = 0.5,
               min_gap: float = 1.0, min_values: int = 5) -> Dict[str, Any]:
    """Where does threshold ``current`` sit in the pilot distribution (log10 scale)?"""
    finite = [float(v) for v in values if v is not None and math.isfinite(v) and v >= 0]
    zeros = sum(v == 0.0 for v in finite)
    pos = sorted(v for v in finite if v > 0)
    lc = math.log10(current)
    below = [v for v in pos if v < current]
    above = [v for v in pos if v >= current]
    m_below = lc - math.log10(below[-1]) if below else math.inf
    m_above = math.log10(above[0]) - lc if above else math.inf
    widest = None
    for a, b in zip(pos, pos[1:]):
        g = math.log10(b) - math.log10(a)
        if widest is None or g > widest["decades"]:
            widest = {"low": a, "high": b, "decades": g, "midpoint": 10 ** ((math.log10(a) + math.log10(b)) / 2)}
    hist: Dict[str, int] = {}
    for v in pos:
        k = str(math.floor(math.log10(v)))
        hist[k] = hist.get(k, 0) + 1
    n = len(finite)
    rep = {"n": n, "n_zero": zeros, "current": current,
           "n_below": len(below) + zeros, "n_above": len(above),
           "nearest_below": below[-1] if below else None, "nearest_above": above[0] if above else None,
           "margin_below_decades": m_below, "margin_above_decades": m_above,
           "widest_gap": widest, "decade_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0])))}
    if n < min_values:
        rep.update(proposed=current, decision="kept_draft", manual_review=True,
                   reason=f"only {n} pilot value(s); too few to calibrate")
    elif m_below >= min_margin and m_above >= min_margin:
        rep.update(proposed=current, decision="kept_draft", manual_review=False,
                   reason=f"no pilot value within {min_margin} decades of the draft value")
    elif widest is not None and widest["decades"] >= min_gap:
        rep.update(proposed=widest["midpoint"], decision="moved_to_gap_midpoint", manual_review=False,
                   reason=(f"draft value has a pilot value within {min_margin} decades; moved to the "
                           f"midpoint of the widest gap ({widest['decades']:.2f} decades)"))
    else:
        rep.update(proposed=current, decision="kept_draft", manual_review=True,
                   reason="pilot values crowd the draft value and no gap of at least "
                          f"{min_gap} decade(s) exists")
    return rep


def calibrate_thresholds(runs: Sequence[Dict[str, Any]], draft: OutcomeThresholds,
                         experiment_name: str = "pilot", **gap_kw: Any) -> Dict[str, Any]:
    """Propose thresholds from ReLU pilot runs. Never reads validation MSE;
    success_val_mse keeps the draft's predetermined value."""
    done = [r for r in runs if r["status"] == "completed"]
    finals = [(r, r["history"][-1]) for r in done]
    reports: Dict[str, Any] = {}

    ratio = [f["shape"]["out_std"] / f["shape"]["target_std"] for _, f in finals]
    reports["collapse_std_frac"] = gap_report(ratio, draft.collapse_std_frac, **gap_kw)
    th = dataclasses.replace(draft, collapse_std_frac=reports["collapse_std_frac"]["proposed"])

    reports["success_val_mse"] = {
        "current": draft.success_val_mse, "proposed": draft.success_val_mse,
        "decision": "fixed_predetermined", "manual_review": False, "n": 0, "n_zero": 0,
        "n_below": None, "n_above": None, "nearest_below": None, "nearest_above": None,
        "margin_below_decades": None, "margin_above_decades": None, "widest_gap": None,
        "decade_histogram": {},
        "reason": ("validation-set threshold, kept at its predetermined value; not calibrated from "
                   "pilot data, and training MSE is not used in its place")}

    live = [f for _, f in finals if not _collapsed(f, th)]
    fit, flat = [], []
    for f in live:
        s = f["shape"]
        good, other = ("left", "right") if s["left_mse"] <= s["right_mse"] else ("right", "left")
        fit.append(s[f"{good}_mse"])
        flat.append(s[f"{other}_out_std"] / s[f"{other}_target_std"])
    reports["half_fit_mse"] = gap_report(fit, draft.half_fit_mse, **gap_kw)
    th = dataclasses.replace(th, half_fit_mse=reports["half_fit_mse"]["proposed"])
    reports["half_flat_std_frac"] = gap_report(flat, draft.half_flat_std_frac, **gap_kw)
    th = dataclasses.replace(th, half_flat_std_frac=reports["half_flat_std_frac"]["proposed"])

    init_range = [r["born"]["out_range"] for r in runs if r.get("born")]
    reports["const_range_tol"] = gap_report(init_range, draft.const_range_tol, **gap_kw)
    th = dataclasses.replace(th, const_range_tol=reports["const_range_tol"]["proposed"])

    quantities = {
        "collapse_std_frac": "final out_std / target_std on the probe grid (all completed runs)",
        "success_val_mse": "none: fixed in advance (compared with validation MSE; not calibrated)",
        "half_fit_mse": "MSE on the better-fitted probe half (all non-collapsed runs)",
        "half_flat_std_frac": "out_std / target_std on the other probe half (same runs)",
        "const_range_tol": "network output range on the training inputs at init (all runs)",
    }
    for k, q in quantities.items():
        reports[k]["quantity"] = q
    proposed = dataclasses.replace(
        th, frozen=False,
        frozen_note=f"PROPOSED from ReLU pilot '{experiment_name}'; review, then freeze with "
                    "scripts/a1_pilot.py freeze")
    return {"draft": draft.to_dict(), "draft_hash": draft.hash(), "proposed": proposed.to_dict(),
            "proposed_hash": proposed.hash(), "reports": reports,
            "manual_review": sorted(k for k, v in reports.items() if v["manual_review"]),
            "changed": sorted(k for k, v in reports.items() if v["proposed"] != v["current"])}


# ----------------------------------------------------------------- summary
def outcome_summary(runs: Sequence[Dict[str, Any]], th: OutcomeThresholds,
                    epochs: Sequence[int] = DEFAULT_CHECKPOINTS) -> Dict[str, Any]:
    """Final outcome counts per arm with exact 95% CIs, plus counts at checkpoints."""
    arms = sorted({r["arm"] for r in runs})
    out: Dict[str, Any] = {"thresholds_hash": th.hash(), "thresholds_frozen": th.frozen, "arms": {}}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        n = len(rs)
        labels = [_final_label(r, th) for r in rs]
        counts = {}
        for o in OUTCOMES:
            k = labels.count(o)
            lo, hi = clopper_pearson(k, n)
            counts[o] = {"count": k, "percent": 100.0 * k / n, "ci95": [100.0 * lo, 100.0 * hi]}
        born = [r["born"]["born_dead"] for r in rs if r.get("born") and r["born"].get("born_dead") is not None]
        kb = sum(born)
        coll_not_born = sum(1 for r, l in zip(rs, labels)
                            if l == "collapsed" and r.get("born") and r["born"].get("born_dead") == 0)
        at = {}
        for e in epochs:
            recs = [record_at(r["history"], e) for r in rs]
            if any(x is None for x in recs):
                continue
            ls = [_label(x, th) for x in recs]
            at[str(e)] = {o: ls.count(o) for o in OUTCOMES if o != "nonfinite"}
        out["arms"][a] = {
            "n": n, "outcomes": counts,
            "born_dead": {"count": kb, "n": len(born),
                          "ci95": [100 * v for v in clopper_pearson(kb, len(born))] if born else None},
            "collapsed_not_born_dead": coll_not_born,
            "outcomes_at_epoch": at,
        }
    return out


# ------------------------------------------------------------------ driver
def _csv(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _g(v: Any, fmt: str = ".3g") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "none"
    return format(v, fmt)


def _report_md(exp: ExperimentConfig, n_runs: int, problems: List[str], budget: Dict[str, Any],
               cal: Dict[str, Any], s_draft: Dict[str, Any], s_prop: Dict[str, Any],
               record: Optional[Dict[str, Any]]) -> str:
    L = [f"# A1 ReLU pilot analysis: {exp.name}", "",
         "**PILOT. Not research results.** ReLU arms only; used to check the training budget and "
         "to calibrate outcome thresholds. No DARL or LeakyReLU comparison is made here.", "",
         f"- Runs analysed: {n_runs} ({', '.join(a.name for a in exp.arms)}, seeds "
         f"{exp.seeds[0]}-{exp.seeds[-1]}, {exp.train['epochs']} epochs)",
         f"- Experiment hash: `{(record or {}).get('experiment_hash', 'n/a')}`, git commit "
         f"`{(record or {}).get('git_commit')}` (dirty: {(record or {}).get('git_dirty')})"]
    if problems:
        L += [f"- **Incomplete:** {len(problems)} run(s) missing or unusable:"] + [f"  - {p}" for p in problems]
    c = budget["criterion"]
    L += ["", "## 1. Training budget (approved criterion)", "",
          f"Change in full-batch training MSE over the last {c['window_frac']:.0%} of epochs must be under "
          f"{c['tol']:.0%} for at least {c['required']:.0%} of non-collapsed runs. Collapse at epoch B uses the "
          "proposed thresholds. Validation loss is not used.", "",
          "| Budget B | Eligible | Converged | Fraction | Passes | Collapsed (excluded) | Median change | Max change |",
          "|---|---|---|---|---|---|---|---|"]
    for r in budget["budgets"]:
        L.append(f"| {r['budget']} | {r['eligible']} | {r['converged']} | {_g(r['fraction'], '.1%')} | "
                 f"{'yes' if r['passes'] else 'no'} | {r['excluded_collapsed']} | {_g(r['median_change'])} | "
                 f"{_g(r['max_change'])} |")
    top = budget["budgets"][-1]
    L += ["", f"**{budget['largest_candidate']} epochs sufficient:** "
          f"{'yes' if budget['largest_candidate_sufficient'] else 'NO'}. "
          f"**Shortest passing budget (all longer budgets also pass):** {_g(budget['recommended_budget'], 'd')}."]
    if top["not_converged"]:
        L += ["", f"Runs not converged at {top['budget']}: " + ", ".join(
            f"{w['arm']} seed {w['seed']} (change {_g(w['change'])}, train MSE {_g(w['train_mse'])})"
            for w in top["not_converged"])]
    L += ["", "## 2. Threshold calibration (proposal; nothing is frozen yet)", "",
          "| Threshold | Draft | Proposed | Decision | Values (zeros) | Nearest below / above | Margin below / above (decades) | Quantity |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in cal["reports"].items():
        L.append(f"| {k} | {_g(r['current'])} | {_g(r['proposed'])} | {r['decision']}"
                 f"{' (MANUAL REVIEW)' if r['manual_review'] else ''} | {r['n']} ({r['n_zero']}) | "
                 f"{_g(r['nearest_below'])} / {_g(r['nearest_above'])} | "
                 f"{_g(r['margin_below_decades'], '.2f')} / {_g(r['margin_above_decades'], '.2f')} | {r['quantity']} |")
    L += ["", "Decade histograms (log10 bin: count; zeros listed separately):", ""]
    for k, r in cal["reports"].items():
        L.append(f"- {k}: zeros {r['n_zero']}; " + ", ".join(f"1e{b}: {n}" for b, n in r["decade_histogram"].items()))
    L += ["", f"Draft hash `{cal['draft_hash'][:12]}`, proposed hash `{cal['proposed_hash'][:12]}`. "
          f"Changed: {', '.join(cal['changed']) or 'none'}. Manual review: {', '.join(cal['manual_review']) or 'none'}."]

    def table(title: str, s: Dict[str, Any]) -> None:
        L.extend(["", title, "", "| Arm | n | " + " | ".join(OUTCOMES) + " | Born dead | Collapsed, not born dead |",
                  "|---|---|" + "---|" * len(OUTCOMES) + "---|---|"])
        for a, v in s["arms"].items():
            cells = [f"{x['count']} ({x['percent']:.0f}%, {x['ci95'][0]:.0f}-{x['ci95'][1]:.0f})"
                     for x in (v["outcomes"][o] for o in OUTCOMES)]
            bd = v["born_dead"]
            L.append(f"| {a} | {v['n']} | " + " | ".join(cells) +
                     f" | {bd['count']}/{bd['n']} | {v['collapsed_not_born_dead']} |")

    L += ["", "## 3. Outcomes (count, %, exact 95% CI)"]
    table("With the PROPOSED thresholds:", s_prop)
    table("With the current DRAFT thresholds:", s_draft)
    L += ["", "Outcome counts at checkpoints (proposed thresholds):", "",
          "| Arm | Epoch | collapsed | half_trained | success | other |", "|---|---|---|---|---|---|"]
    for a, v in s_prop["arms"].items():
        for e, cnt in v["outcomes_at_epoch"].items():
            L.append(f"| {a} | {e} | {cnt['collapsed']} | {cnt['half_trained']} | {cnt['success']} | {cnt['other']} |")
    L += ["", "With 20 seeds per arm the intervals are wide; these rates are not estimates for the paper.",
          "", "## 4. Paper definitions vs numerical proxies", "",
          "| Quantity | Paper (theoretical) definition | Numerical proxy used here |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c_} |" for a, b, c_ in PROXY_VS_PAPER]
    L += ["", "Files: per_seed.csv, checkpoints.csv, budget.json, calibration.json, outcomes_proposed.json, "
          "summary.json.", ""]
    return "\n".join(L)


def analyze_pilot(exp: ExperimentConfig, draft: OutcomeThresholds, allow_incomplete: bool = False,
                  out_dir: Optional[Path] = None, **gap_kw: Any) -> Dict[str, Any]:
    """Run the full pilot analysis and write it to ``<root>/analysis``. Returns the summary."""
    runs, problems = load_pilot_runs(exp, draft, allow_incomplete=allow_incomplete)
    if not runs:
        raise ValueError("no finished pilot runs to analyse")
    cal = calibrate_thresholds(runs, draft, exp.name, **gap_kw)
    proposed = OutcomeThresholds(**cal["proposed"])
    budget = budget_check(runs, proposed)
    s_draft = outcome_summary(runs, draft)
    s_prop = outcome_summary(runs, proposed)
    out_dir = Path(out_dir) if out_dir is not None else exp.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = _read_json(exp.root / "experiment.json")
    summary = {"experiment": exp.name, "mode": exp.mode, "note": "PILOT - not research results",
               "n_runs": len(runs), "incomplete": problems, "budget": budget,
               "calibration": {k: cal[k] for k in ("draft_hash", "proposed_hash", "changed", "manual_review")},
               "outcomes_draft": s_draft, "outcomes_proposed": s_prop,
               "experiment_hash": (record or {}).get("experiment_hash")}
    atomic_write_text(out_dir / "per_seed.csv", _csv(seed_table(runs, proposed)))
    atomic_write_text(out_dir / "checkpoints.csv", _csv(checkpoint_table(runs, proposed)))
    atomic_write_text(out_dir / "budget.json", json.dumps(budget, indent=2))
    atomic_write_text(out_dir / "calibration.json", json.dumps(cal, indent=2))
    write_thresholds(out_dir / "outcomes_proposed.json", proposed)
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2))
    atomic_write_text(out_dir / "pilot_report.md",
                      _report_md(exp, len(runs), problems, budget, cal, s_draft, s_prop, record))
    return summary


def freeze_thresholds(proposed_path: Path, target: Path, note: str) -> OutcomeThresholds:
    """Write the reviewed proposal to ``target`` with frozen=true. Refuses to overwrite a frozen file."""
    if not note.strip():
        raise ValueError("a freeze note is required (who reviewed what, and from which pilot)")
    raw = json.loads(Path(proposed_path).read_text(encoding="utf-8"))
    th = OutcomeThresholds(**raw)
    if Path(target).exists():
        cur = _read_json(Path(target)) or {}
        if cur.get("frozen"):
            raise ValueError(f"{target} is already frozen; thresholds are never re-frozen in place")
    frozen = dataclasses.replace(th, frozen=True,
                                 frozen_note=f"{note.strip()} (frozen {_dt.date.today().isoformat()})")
    write_thresholds(target, frozen)
    return frozen
