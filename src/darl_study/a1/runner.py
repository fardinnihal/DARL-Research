"""Experiment runner for A1: many (arm, seed) training runs, resumable and paired.

An experiment config (JSON, e.g. ``configs/a1/smoke.json``) names a mode, a
seed range, shared training settings and a list of arms (activation + init).
Every arm runs on the same seeds; run seed s gives the same dataset, the same
initial weight tensors (Step 3) and the same batch order (Step 4) in every
arm, so arms are paired by seed.

Folders (all under results/, which is git-ignored)::

    results/a1/<mode>/<name>/experiment.json        resolved experiment record
    results/a1/<mode>/<name>/runs_index.csv         one row per finished run
    results/a1/<mode>/<name>/<arm>/seed_0000/
        config.json     training config, hash, seeds, dataset hashes (train())
        checkpoint.pt   latest state, atomic (train())
        result.json     status + full metric history (train())
        outcome.json    born-dead labels at init + outcome classification
        DONE            training-config hash + thresholds hash, written last

Skip / resume rules
-------------------
* Skipped only if result.json, outcome.json and DONE are all present, parse,
  and agree with each other and with the current training-config hash and
  thresholds hash.
* result.json valid and terminal, thresholds changed -> outcome is
  re-classified from result.json; training is NOT rerun.
* result.json valid and terminal, but outcome.json or DONE missing, corrupt
  or inconsistent -> outcome regenerated from result.json; training is NOT
  rerun.
* Folder belongs to a different training config -> error (never overwritten).
* No valid terminal result.json -> ``train()`` runs and resumes from
  checkpoint.pt if present.
* A nonfinite run is terminal: recorded, classified "nonfinite", never retried.

Modes and thresholds
--------------------
mode -> outcome purpose: smoke -> pilot, pilot -> pilot,
calibration -> calibration, main -> main. Draft thresholds are allowed for
smoke, pilot and calibration; a main experiment refuses them before any
training starts.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import multiprocessing as mp
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch

from darl_study.a1.checkpoint import atomic_write_text, config_hash, load_checkpoint
from darl_study.a1.data import build_a1_datasets, load_data_config
from darl_study.a1.init import initialize_
from darl_study.a1.metrics import a1_extra_metrics, recovery_summary
from darl_study.a1.model import DeepNarrowMLP
from darl_study.a1.outcomes import (
    OutcomeThresholds,
    OUTCOMES,
    born_dead,
    check_thresholds,
    classify_result,
    load_thresholds,
)
from darl_study.a1.train import CONFIG_NAME, RESULT_NAME, TrainConfig, train

__all__ = [
    "MODES",
    "MODE_PURPOSE",
    "Arm",
    "ExperimentConfig",
    "load_experiment",
    "run_dir_for",
    "run_status",
    "run_one",
    "run_experiment",
]

REPO = Path(__file__).resolve().parents[3]
MODES = ("smoke", "pilot", "calibration", "main")
MODE_PURPOSE = {"smoke": "pilot", "pilot": "pilot", "calibration": "calibration", "main": "main"}
DONE_NAME = "DONE"
OUTCOME_NAME = "outcome.json"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# TrainConfig fields an experiment may set for all arms.
_SHARED_TRAIN_KEYS = {f.name for f in dataclasses.fields(TrainConfig)} - {
    "activation", "act_kwargs", "init", "run_seed", "data"}


# ------------------------------------------------------------------ config
@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    activation: str
    init: str
    act_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE.match(self.name):
            raise ValueError(f"arm name {self.name!r} must be 1-64 chars of letters, digits, . _ -")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    name: str
    mode: str
    seed_start: int
    seed_count: int
    arms: Sequence[Arm]
    train: Dict[str, Any]
    data_config: str = "configs/a1/data.json"
    thresholds: str = "configs/a1/outcomes.json"
    results_root: str = "results/a1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE.match(self.name):
            raise ValueError(f"experiment name {self.name!r} is not a safe folder name")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for k, v in (("seed_start", self.seed_start), ("seed_count", self.seed_count)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{k} must be an int")
        if self.seed_start < 0 or self.seed_count < 1:
            raise ValueError("need seed_start >= 0 and seed_count >= 1")
        if not self.arms:
            raise ValueError("at least one arm is required")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"arm names must be unique: {names}")
        unknown = set(self.train) - _SHARED_TRAIN_KEYS
        if unknown:
            raise ValueError(f"unknown or per-arm train keys: {sorted(unknown)}")
        for a in self.arms:  # validates activation, init and arguments now, not mid-run
            self.train_config(a, self.seed_start, load_data_config(self.path(self.data_config)))

    @property
    def purpose(self) -> str:
        return MODE_PURPOSE[self.mode]

    @property
    def seeds(self) -> List[int]:
        return list(range(self.seed_start, self.seed_start + self.seed_count))

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else REPO / q

    @property
    def root(self) -> Path:
        return self.path(self.results_root) / self.mode / self.name

    def train_config(self, arm: Arm, seed: int, data_cfg) -> TrainConfig:
        return TrainConfig(activation=arm.activation, act_kwargs=dict(arm.act_kwargs), init=arm.init,
                           run_seed=seed, data=data_cfg, **self.train)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["arms"] = [dataclasses.asdict(a) for a in self.arms]
        return d


def load_experiment(path: Union[str, Path], **overrides: Any) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"name", "mode", "seeds", "arms", "train", "data_config", "thresholds", "results_root", "comment"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
    seeds = raw.get("seeds", {})
    if set(seeds) - {"start", "count"}:
        raise ValueError("seeds must be {'start': int, 'count': int}")
    arms = []
    for a in raw.get("arms", []):
        if set(a) - {"name", "activation", "init", "act_kwargs"}:
            raise ValueError(f"unknown arm keys in {a}")
        arms.append(Arm(a["name"], a["activation"], a["init"], dict(a.get("act_kwargs", {}))))
    kw = dict(name=raw.get("name"), mode=raw.get("mode"), seed_start=seeds.get("start", 0),
              seed_count=seeds.get("count"), arms=tuple(arms), train=dict(raw.get("train", {})))
    for k in ("data_config", "thresholds", "results_root"):
        if k in raw:
            kw[k] = raw[k]
    kw.update(overrides)
    return ExperimentConfig(**kw)


def run_dir_for(exp: ExperimentConfig, arm: Arm, seed: int) -> Path:
    return exp.root / arm.name / f"seed_{seed:04d}"


# ------------------------------------------------------------- one run
def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    """Parsed JSON object, or None if the file is missing, unreadable or not a JSON object."""
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):  # JSONDecodeError is a ValueError
        return None
    return obj if isinstance(obj, dict) else None


_TERMINAL = ("completed", "nonfinite")


def _valid_result(res: Optional[Dict[str, Any]], cfg_hash: str) -> bool:
    """result.json is a finished (terminal) run of this exact training config."""
    if res is None or res.get("config_hash") != cfg_hash or res.get("status") not in _TERMINAL:
        return False
    hist = res.get("history")
    if not isinstance(hist, list):
        return False
    if res["status"] == "completed":
        epochs = (res.get("config") or {}).get("epochs")
        return bool(hist) and epochs is not None and res.get("epochs_completed") == epochs
    return True


def _valid_outcome(out: Optional[Dict[str, Any]], done: Optional[Dict[str, Any]],
                   res: Dict[str, Any], cfg_hash: str, th_hash: str) -> bool:
    """outcome.json and DONE both exist and agree with result.json and both hashes."""
    if out is None or done is None:
        return False
    if out.get("outcome") not in OUTCOMES:
        return False
    for rec in (out, done):
        if (rec.get("config_hash") != cfg_hash or rec.get("thresholds_hash") != th_hash
                or rec.get("status") != res["status"]):
            return False
    if done.get("outcome") != out["outcome"]:
        return False
    return (out["outcome"] == "nonfinite") == (res["status"] == "nonfinite")


def run_status(run_dir: Path, cfg_hash: str, th_hash: str) -> str:
    """'done', 'reclassify', 'regenerate', 'conflict', 'partial' or 'new'.

    * conflict    config.json or DONE belongs to a different training config
    * done        result.json, outcome.json and DONE valid and consistent
    * reclassify  valid terminal result, DONE from other thresholds
    * regenerate  valid terminal result, outcome.json/DONE missing, corrupt or
                  inconsistent (outcome is rebuilt from result.json, no training)
    * partial     no valid terminal result, but a checkpoint or DONE exists
    * new         nothing usable
    """
    done = _read_json(run_dir / DONE_NAME)
    cfg_rec = _read_json(run_dir / CONFIG_NAME)
    if cfg_rec is not None and cfg_rec.get("config_hash") != cfg_hash:
        return "conflict"
    if done is not None and done.get("config_hash") != cfg_hash:
        return "conflict"
    res = _read_json(run_dir / RESULT_NAME)
    if not _valid_result(res, cfg_hash):
        # No verified finished result: (re)train, resuming from the checkpoint if any.
        return "partial" if (done is not None or (run_dir / "checkpoint.pt").exists()) else "new"
    out = _read_json(run_dir / OUTCOME_NAME)
    if _valid_outcome(out, done, res, cfg_hash, th_hash):
        return "done"
    if (done is not None and done.get("thresholds_hash") != th_hash
            and done.get("status") == res["status"]):
        return "reclassify"
    return "regenerate"


def _outcome(result: Dict[str, Any], th: OutcomeThresholds, born: Optional[Dict[str, Any]],
             purpose: str) -> Dict[str, Any]:
    rec = classify_result(result, th, born, purpose=purpose)
    hist = result.get("history") or []
    if hist and "neurons" in hist[0]:
        rec["recovery"] = recovery_summary([h["neurons"]["q"] for h in hist],
                                           [h["neurons"]["z_range"] for h in hist])
        rec["dead_units_at_init"] = hist[0]["neurons"]["dead_count"]
        rec["dead_units_final"] = hist[-1]["neurons"]["dead_count"]
    rec["born_dead_record"] = born  # None if the initial network output was non-finite
    rec["darl_alpha_time_mean"] = result.get("darl_alpha_time_mean")
    rec["failure"] = result.get("failure")
    return rec


def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run, resume, skip, re-classify or regenerate the outcome of one (arm, seed). Picklable for worker processes."""
    torch.set_num_threads(int(job.get("threads", 1)))
    cfg = TrainConfig.from_dict(job["train_config"])
    th = OutcomeThresholds(**job["thresholds"])
    purpose = job["purpose"]
    check_thresholds(th, purpose)
    run_dir = Path(job["run_dir"])
    chash, thash = cfg.hash(), th.hash()
    t0 = time.perf_counter()
    status = run_status(run_dir, chash, thash)
    base = {"arm": job["arm"], "seed": cfg.run_seed, "run_dir": str(run_dir)}
    if status == "conflict":
        return {**base, "action": "conflict", "status": None, "outcome": None,
                "message": f"{run_dir} holds a run with a different training config; not touched"}
    if status == "done":
        out = _read_json(run_dir / OUTCOME_NAME) or {}
        return {**base, "action": "skipped", "status": out.get("status"), "outcome": out.get("outcome"),
                "seconds": 0.0}

    data = build_a1_datasets(cfg.data)
    model = DeepNarrowMLP(cfg.n_layers, cfg.width, 1, 1, cfg.activation, cfg.act_kwargs)
    initialize_(model, cfg.init, cfg.run_seed)  # same init as train(): same seed, same tensors
    try:
        born: Optional[Dict[str, Any]] = born_dead(model, data["train"].x, th.const_range_tol)
    except FloatingPointError:
        born = None  # non-finite initial output; train() records the run as nonfinite

    if status in ("reclassify", "regenerate"):
        result = _read_json(run_dir / RESULT_NAME)  # validated by run_status; no training
        action = "reclassified" if status == "reclassify" else "regenerated"
    else:
        resumed_from = None
        if status == "partial":
            ck = run_dir / "checkpoint.pt"
            if ck.exists():
                resumed_from = load_checkpoint(ck, expected_config_hash=chash)["run"]["epoch"]
        result = train(cfg, run_dir=run_dir, extra_metrics=a1_extra_metrics)
        action = "trained" if resumed_from is None else f"resumed@{resumed_from}"
    out = _outcome(result, th, born, purpose)
    out.update({"arm": job["arm"], "seed": cfg.run_seed, "config_hash": chash, "mode": job["mode"]})
    atomic_write_text(run_dir / OUTCOME_NAME, json.dumps(out, indent=2))
    atomic_write_text(run_dir / DONE_NAME, json.dumps(
        {"config_hash": chash, "thresholds_hash": thash, "status": result["status"],
         "outcome": out["outcome"]}, indent=2))
    return {**base, "action": action, "status": result["status"], "outcome": out["outcome"],
            "seconds": time.perf_counter() - t0, "failure": result.get("failure")}


# ---------------------------------------------------------- experiment
def _git_info() -> Dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                               text=True, timeout=10).stdout.strip() != ""
        return {"git_commit": commit, "git_dirty": dirty if commit else None}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _jobs(exp: ExperimentConfig, th: OutcomeThresholds, arms: Sequence[Arm], threads: int) -> List[Dict[str, Any]]:
    data_cfg = load_data_config(exp.path(exp.data_config))
    jobs = []
    for seed in exp.seeds:          # seed-major order keeps paired arms close in time
        for arm in arms:
            cfg = exp.train_config(arm, seed, data_cfg)
            jobs.append({"arm": arm.name, "train_config": cfg.to_dict(), "thresholds": th.to_dict(),
                         "purpose": exp.purpose, "mode": exp.mode, "threads": threads,
                         "run_dir": str(run_dir_for(exp, arm, seed))})
    return jobs


def _write_experiment_record(exp: ExperimentConfig, th: OutcomeThresholds) -> None:
    exp.root.mkdir(parents=True, exist_ok=True)
    data_cfg = load_data_config(exp.path(exp.data_config))
    data = build_a1_datasets(data_cfg)
    record = {
        "experiment": exp.to_dict(),
        "purpose": exp.purpose,
        "experiment_hash": config_hash({k: v for k, v in exp.to_dict().items()
                                        if k not in ("seed_count",)}),
        "thresholds": th.to_dict(),
        "thresholds_hash": th.hash(),
        "dataset_sha256": {k: v.fingerprint() for k, v in data.items()},
        "note": ("SMOKE TEST: pipeline check only; not research results."
                 if exp.mode == "smoke" else None),
        **_git_info(),
    }
    path = exp.root / "experiment.json"
    old = _read_json(path)
    if old is not None and old.get("experiment_hash") != record["experiment_hash"]:
        raise ValueError(f"{exp.root} already holds a different experiment "
                         f"(hash {old.get('experiment_hash', '?')[:12]}); use a new name")
    atomic_write_text(path, json.dumps(record, indent=2))


def _index(exp: ExperimentConfig) -> List[Dict[str, Any]]:
    rows = []
    for arm in exp.arms:
        for seed in exp.seeds:
            out = _read_json(run_dir_for(exp, arm, seed) / OUTCOME_NAME)
            if out is None:
                continue
            rec = out.get("recovery") or {}
            born = out.get("born_dead_record") or {}
            rows.append({
                "arm": arm.name, "activation": arm.activation, "init": arm.init, "seed": seed,
                "status": out.get("status"), "outcome": out.get("outcome"),
                "born_dead": born.get("born_dead"),
                "structural_dead_layer_at_init": born.get("structural_dead_layer"),
                "collapsed_at_init": out.get("collapsed_at_init"),
                "collapsed_at_end": out.get("collapsed_at_end"),
                "first_collapse_epoch": out.get("first_collapse_epoch"),
                "final_val_mse": out.get("final_val_mse"),
                "final_out_std": out.get("final_out_std"),
                "dead_units_at_init": out.get("dead_units_at_init"),
                "dead_units_final": out.get("dead_units_final"),
                "recovery_eligible": rec.get("recovery_eligible"),
                "recovered": rec.get("recovered"),
                "darl_alpha_time_mean": out.get("darl_alpha_time_mean"),
                "thresholds_frozen": out.get("thresholds_frozen"),
                "purpose": out.get("purpose"),
            })
    return rows


def _write_index(exp: ExperimentConfig, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(exp.root / "runs_index.csv", buf.getvalue())


def run_experiment(
    exp: ExperimentConfig,
    arm_names: Optional[Iterable[str]] = None,
    workers: int = 1,
    threads: int = 1,
    report: Callable[[str], None] = print,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run every (arm, seed) of the experiment. Returns one summary per run."""
    th = load_thresholds(exp.path(exp.thresholds), purpose=exp.purpose)  # main refuses drafts here
    arms = list(exp.arms)
    if arm_names is not None:
        wanted = list(arm_names)
        missing = set(wanted) - {a.name for a in arms}
        if missing:
            raise ValueError(f"unknown arms {sorted(missing)}; have {[a.name for a in arms]}")
        arms = [a for a in arms if a.name in wanted]
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be >= 1")
    jobs = _jobs(exp, th, arms, threads)

    banner = (f"A1 {exp.mode.upper()} '{exp.name}': {len(arms)} arms x {exp.seed_count} seeds "
              f"= {len(jobs)} runs, epochs={jobs[0]['train_config']['epochs']}, "
              f"thresholds {'FROZEN' if th.frozen else 'DRAFT'} (purpose {exp.purpose})")
    report(banner)
    if exp.mode == "smoke":
        report("SMOKE TEST: checks the pipeline only. Do not draw scientific conclusions from it.")
    report(f"output: {exp.root}")

    if dry_run:
        out = []
        for j in jobs:
            cfg = TrainConfig.from_dict(j["train_config"])
            st = run_status(Path(j["run_dir"]), cfg.hash(), th.hash())
            out.append({"arm": j["arm"], "seed": cfg.run_seed, "run_dir": j["run_dir"], "action": f"would:{st}"})
            report(f"  {j['arm']:<14} seed {cfg.run_seed:04d}  {st}")
        return out

    _write_experiment_record(exp, th)
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    def log(r: Dict[str, Any]) -> None:
        results.append(r)
        extra = f"  {r['failure']['phase']}: {r['failure']['message']}" if r.get("failure") else ""
        msg = r.get("message", "")
        report(f"[{len(results):>4}/{len(jobs)}] {r['arm']:<14} seed {r['seed']:04d}  "
               f"{r['action']:<12} {str(r['status']):<10} {str(r['outcome']):<12} "
               f"{r.get('seconds', 0.0):6.1f}s{extra}{msg}")

    try:
        if workers == 1:
            for j in jobs:
                log(run_one(j))
        else:
            ctx = mp.get_context("spawn")  # Windows-safe, and the same on Linux
            with ctx.Pool(processes=workers) as pool:
                for r in pool.imap_unordered(run_one, jobs):
                    log(r)
    finally:
        _write_index(exp, _index(exp))

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["action"].split("@")[0]] = counts.get(r["action"].split("@")[0], 0) + 1
    report(f"done in {time.perf_counter() - t0:.1f}s: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    conflicts = [r for r in results if r["action"] == "conflict"]
    if conflicts:
        report(f"WARNING: {len(conflicts)} run folder(s) belong to a different config and were left untouched.")
    return results
