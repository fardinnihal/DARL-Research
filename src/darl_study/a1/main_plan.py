"""A1 Step 9: slope-matched LeakyReLU calibration and the main comparison config.

Approved design (A1 Step 1 plan, section 4; roadmap decision D7):

* Main arms, paired by seed (same data, initial tensors and batch order):
  ReLU, LeakyReLU 0.01, DARL (approved defaults) and LeakyReLU at the
  slope-matched value, each with He (primary set) and RAI (secondary set).
* The matched slope is DARL's mean negative slope over all 18 hidden units
  and all training steps (``darl_alpha_time_mean`` of each run), averaged over
  a separate set of DARL calibration seeds. One number per initialization,
  written into the main config before any main run starts. Calibration seeds
  are never reused in the main experiment, and only training-time DARL state
  enters the number: no validation, probe or test quantity is used.

Flow:

    python scripts/a1_run.py --config configs/a1/calibration.json --workers 3
    python scripts/a1_slope_match.py                       # preview, writes nothing
    python scripts/a1_slope_match.py --write               # writes configs/a1/main.json
    python scripts/a1_run.py --config configs/a1/main.json --dry-run

``configs/a1/main.json`` does not exist until calibration is finished, so the
main experiment cannot be started with a guessed slope. The template's
placeholders (``"<slope_match:he>"``) are not numbers, so loading the template
itself as an experiment fails.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from darl_study.a1.checkpoint import config_hash
from darl_study.a1.data import load_data_config
from darl_study.a1.model import DARL_A1_DEFAULTS
from darl_study.a1.outcomes import load_thresholds
from darl_study.a1.runner import (
    OUTCOME_NAME,
    ExperimentConfig,
    _read_json,
    experiment_from_dict,
    load_experiment,
    run_dir_for,
    run_status,
)
from darl_study.a1.train import RESULT_NAME

__all__ = [
    "SLOPE_RULE",
    "PLACEHOLDER",
    "experiment_hash",
    "load_calibration_runs",
    "slope_match",
    "fill_template",
    "check_seeds_disjoint",
    "build_main_config",
]

SLOPE_RULE = ("alpha_bar(init) = mean over completed calibration runs of that init of "
              "darl_alpha_time_mean (mean over all 18 hidden units and all training steps "
              "of DARL's negative slope); nonfinite runs excluded and counted")
PLACEHOLDER = re.compile(r"^<slope_match:([a-z]+)>$")
TEMPLATE_KEY = "slope_match"


def experiment_hash(exp: ExperimentConfig) -> str:
    """Same hash the runner writes to experiment.json (seed_count excluded)."""
    return config_hash({k: v for k, v in exp.to_dict().items() if k != "seed_count"})


# ------------------------------------------------------------ calibration
def _check_calibration_design(exp: ExperimentConfig) -> None:
    if exp.mode != "calibration":
        raise ValueError(f"slope matching needs a calibration experiment, got mode {exp.mode!r}")
    inits = [a.init for a in exp.arms]
    if len(set(inits)) != len(inits):
        raise ValueError(f"one DARL arm per initialization expected, got inits {inits}")
    for a in exp.arms:
        if a.activation != "darl":
            raise ValueError(f"calibration arms must be DARL, {a.name} is {a.activation}")
        resolved = {**DARL_A1_DEFAULTS, **a.act_kwargs}
        if resolved != DARL_A1_DEFAULTS:
            raise ValueError(f"{a.name}: calibration must use the approved DARL defaults "
                             f"{DARL_A1_DEFAULTS}, got {resolved}")


def load_calibration_runs(exp: ExperimentConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Finished calibration runs per initialization. Refuses anything incomplete.

    Every run must be 'done' for this config and the experiment's thresholds
    file (result.json, outcome.json and DONE valid and consistent), and the
    experiment record must match this config.
    """
    _check_calibration_design(exp)
    th = load_thresholds(exp.path(exp.thresholds), purpose=exp.purpose)
    record = _read_json(exp.root / "experiment.json")
    if record is None:
        raise ValueError(f"{exp.root / 'experiment.json'} is missing; run the calibration first")
    if record.get("experiment_hash") != experiment_hash(exp):
        raise ValueError(f"{exp.root} holds a different experiment (experiment hash mismatch)")
    if record.get("thresholds_hash") != th.hash():
        raise ValueError("calibration was classified with other thresholds; rerun a1_run.py on it first")
    data_cfg = load_data_config(exp.path(exp.data_config))
    out: Dict[str, List[Dict[str, Any]]] = {}
    problems: List[str] = []
    for arm in exp.arms:
        rows = []
        for seed in exp.seeds:
            d = run_dir_for(exp, arm, seed)
            chash = exp.train_config(arm, seed, data_cfg).hash()
            st = run_status(d, chash, th.hash())
            if st != "done":
                problems.append(f"{arm.name} seed {seed}: status {st}")
                continue
            res = _read_json(d / RESULT_NAME)
            o = _read_json(d / OUTCOME_NAME)
            final = res.get("final") or {}
            rows.append({"seed": seed, "status": res["status"], "outcome": o["outcome"],
                         "alpha_time_mean": res.get("darl_alpha_time_mean"),
                         "alpha_final_mean": final.get("darl_alpha_mean"),
                         "darl_num_updates": res.get("darl_num_updates"),
                         "steps_completed": res.get("steps_completed")})
        out[arm.init] = rows
    if problems:
        raise ValueError(f"{len(problems)} calibration run(s) not finished:\n  " + "\n  ".join(problems[:20]))
    return out


def _stats(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    m = math.fsum(xs) / n
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": m, "sd": sd, "min": min(xs), "max": max(xs)}


def slope_match(exp: ExperimentConfig) -> Dict[str, Any]:
    """The slope-matched LeakyReLU value per initialization, with full provenance."""
    runs = load_calibration_runs(exp)
    per_init = {}
    for init, rows in runs.items():
        done = [r for r in rows if r["status"] == "completed"]
        for r in done:
            steps, upd = r["steps_completed"], r["darl_num_updates"]
            if not (isinstance(upd, list) and upd and all(u == steps for u in upd)):
                raise ValueError(f"{init} seed {r['seed']}: DARL update count {upd} != steps {steps}")
            a = r["alpha_time_mean"]
            if not (isinstance(a, (int, float)) and math.isfinite(a)
                    and DARL_A1_DEFAULTS["alpha_min"] <= a <= DARL_A1_DEFAULTS["alpha_max"]):
                raise ValueError(f"{init} seed {r['seed']}: alpha time mean {a!r} outside "
                                 "[alpha_min, alpha_max]")
        if not done:
            raise ValueError(f"no completed calibration run for init {init!r}")
        time_means = [r["alpha_time_mean"] for r in done]
        st = _stats(time_means)
        outcomes: Dict[str, int] = {}
        for r in rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        per_init[init] = {
            "alpha_bar": st["mean"],
            "time_mean_stats": st,
            "final_mean_stats_reported_only": _stats([r["alpha_final_mean"] for r in done
                                                      if r["alpha_final_mean"] is not None]),
            "n_runs": len(rows),
            "n_completed": len(done),
            "n_nonfinite_excluded": len(rows) - len(done),
            "calibration_outcomes": outcomes,
            "per_seed": [{"seed": r["seed"], "status": r["status"], "outcome": r["outcome"],
                          "alpha_time_mean": r["alpha_time_mean"]} for r in rows],
        }
    return {
        "rule": SLOPE_RULE,
        "calibration_experiment": exp.name,
        "calibration_experiment_hash": experiment_hash(exp),
        "calibration_seeds": {"start": exp.seed_start, "count": exp.seed_count},
        "darl_parameters": dict(DARL_A1_DEFAULTS),
        "epochs": exp.train.get("epochs"),
        "per_init": per_init,
        "created": _dt.date.today().isoformat(),
    }


# --------------------------------------------------------------- template
def _seed_range(start: int, count: int) -> Tuple[int, int]:
    return start, start + count  # half-open


def check_seeds_disjoint(main: ExperimentConfig, used: Sequence[Tuple[str, int, int]]) -> None:
    """Refuse a main experiment whose seeds overlap any (label, start, count) in ``used``."""
    a0, a1 = _seed_range(main.seed_start, main.seed_count)
    for label, s, c in used:
        b0, b1 = _seed_range(s, c)
        if a0 < b1 and b0 < a1:
            raise ValueError(f"main seeds {a0}-{a1 - 1} overlap {label} seeds {b0}-{b1 - 1}; "
                             "pilot and calibration seeds are never reused")


def fill_template(template: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    """Main config dict with every ``<slope_match:init>`` replaced by alpha_bar(init)."""
    t = json.loads(json.dumps(template))  # deep copy
    t.pop(TEMPLATE_KEY, None)
    if t.get("mode") != "main":
        raise ValueError(f"the main template must have mode 'main', got {t.get('mode')!r}")
    filled = 0
    for arm in t.get("arms", []):
        for k, v in list(arm.get("act_kwargs", {}).items()):
            if isinstance(v, str):
                m = PLACEHOLDER.match(v)
                if not m:
                    raise ValueError(f"arm {arm.get('name')}: {k}={v!r} is not a number or a placeholder")
                init = m.group(1)
                if init != arm.get("init"):
                    raise ValueError(f"arm {arm.get('name')} ({arm.get('init')}) uses the {init} slope")
                if init not in match["per_init"]:
                    raise ValueError(f"no calibrated slope for init {init!r}")
                arm["act_kwargs"][k] = match["per_init"][init]["alpha_bar"]
                filled += 1
    if filled == 0:
        raise ValueError("the template has no slope placeholders")
    t["comment"] = (str(t.get("comment", "")).strip() + " | slope-matched LeakyReLU from calibration "
                    f"'{match['calibration_experiment']}' (hash {match['calibration_experiment_hash'][:12]}, "
                    f"seeds {match['calibration_seeds']['start']}-"
                    f"{match['calibration_seeds']['start'] + match['calibration_seeds']['count'] - 1}): "
                    + ", ".join(f"{i} {v['alpha_bar']:.6f}" for i, v in sorted(match["per_init"].items()))
                    + f"; written {match['created']}").strip(" |")
    return t


def build_main_config(calibration_cfg: Union[str, Path], template_path: Union[str, Path],
                      pilot_cfg: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(main config dict, slope-match record). Validates everything; writes nothing."""
    cal = load_experiment(calibration_cfg)
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    ref = template.get(TEMPLATE_KEY, {}).get("calibration_config")
    if ref is not None and Path(cal.path(ref)).resolve() != Path(calibration_cfg).resolve():
        raise ValueError(f"the template expects calibration config {ref}, got {calibration_cfg}")
    match = slope_match(cal)
    main_dict = fill_template(template, match)
    main = experiment_from_dict(main_dict)
    used = [(f"calibration '{cal.name}'", cal.seed_start, cal.seed_count)]
    if pilot_cfg is not None:
        p = load_experiment(pilot_cfg)
        used.append((f"pilot '{p.name}'", p.seed_start, p.seed_count))
    check_seeds_disjoint(main, used)
    for k in ("train", "data_config", "results_root"):
        if getattr(main, k) != getattr(cal, k):
            raise ValueError(f"main and calibration differ in {k}; the slope would not be matched")
    th = load_thresholds(main.path(main.thresholds), purpose="main")  # refuses draft thresholds
    match["main_thresholds_hash"] = th.hash()
    match["main_experiment_hash"] = experiment_hash(main)
    return main_dict, match

