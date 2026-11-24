"""Fingerprint a finished A1 experiment, or compare two fingerprints (e.g. two machines).

    python scripts/a1_fingerprint.py make --config configs/a1/pilot.json --out pilot_fingerprint.json
    python scripts/a1_fingerprint.py compare local.json cloud.json

``make`` only reads. It records, per run: the training-config hash, seeds,
software versions, status, epoch/step counts, the SHA-256 of result.json and
every recorded full-batch training MSE as an exact float (hex), plus the
experiment record (experiment hash, thresholds hash, dataset SHA-256, git
commit), the budget criterion of any analysis folder, and the SHA-256 of the
A1 source files. ``compare`` reports, for each run, whether the inputs agree
(config hash, dataset, seeds) and the first epoch at which the saved training
MSE stops being bit-identical, so cross-machine differences can be located
exactly instead of assumed harmless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "darl_study"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def make(config: Path) -> Dict[str, Any]:
    from darl_study.a1.runner import load_experiment, run_dir_for

    exp = load_experiment(config)
    rec = _read(exp.root / "experiment.json") or {}
    analyses = {}
    for d in sorted(exp.root.glob("analysis*")):
        b = _read(d / "budget.json")
        if b is not None:
            analyses[d.name] = {"criterion": b.get("criterion"),
                                "recommended_budget": b.get("recommended_budget"),
                                "largest_candidate_sufficient": b.get("largest_candidate_sufficient")}
    runs: Dict[str, Any] = {}
    for arm in exp.arms:
        for seed in exp.seeds:
            d = run_dir_for(exp, arm, seed)
            res = _read(d / "result.json")
            if res is None:
                runs[f"{arm.name}/{seed}"] = None
                continue
            h = res.get("history") or []
            runs[f"{arm.name}/{seed}"] = {
                "config_hash": res.get("config_hash"), "seeds": res.get("seeds"),
                "versions": res.get("versions"), "status": res.get("status"),
                "epochs_completed": res.get("epochs_completed"), "steps_completed": res.get("steps_completed"),
                "result_sha256": _sha(d / "result.json"),
                "epochs": [r["epoch"] for r in h],
                "train_mse_hex": [float(r["train_mse"]).hex() for r in h],
            }
    return {"experiment": exp.name, "mode": exp.mode,
            "experiment_record": {k: rec.get(k) for k in ("experiment_hash", "thresholds_hash", "dataset_sha256",
                                                          "git_commit", "git_dirty")},
            "analyses": analyses,
            "source_sha256": {str(p.relative_to(SRC)).replace("\\", "/"): _sha(p)
                              for p in sorted((SRC / "a1").glob("*.py")) + sorted(SRC.glob("*.py"))},
            "runs": runs}


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ea, eb = a["experiment_record"], b["experiment_record"]
    out: Dict[str, Any] = {
        "dataset_sha256_equal": ea.get("dataset_sha256") == eb.get("dataset_sha256"),
        "thresholds_hash_equal": ea.get("thresholds_hash") == eb.get("thresholds_hash"),
        "experiment_hash_equal": ea.get("experiment_hash") == eb.get("experiment_hash"),
        "git_commit": [ea.get("git_commit"), eb.get("git_commit")],
        "source_files_differing": sorted(k for k in set(a["source_sha256"]) | set(b["source_sha256"])
                                         if a["source_sha256"].get(k) != b["source_sha256"].get(k)),
        "analysis_criteria_equal": {k: (a["analyses"].get(k) or {}).get("criterion") ==
                                    (b["analyses"].get(k) or {}).get("criterion")
                                    for k in sorted(set(a["analyses"]) | set(b["analyses"]))},
        "runs": {},
    }
    for key in sorted(set(a["runs"]) | set(b["runs"]), key=lambda k: (k.split("/")[0], int(k.split("/")[1]))):
        ra, rb = a["runs"].get(key), b["runs"].get(key)
        if ra is None or rb is None:
            out["runs"][key] = {"missing": [ra is None, rb is None]}
            continue
        ha, hb = ra["train_mse_hex"], rb["train_mse_hex"]
        first = next((ra["epochs"][i] for i in range(min(len(ha), len(hb))) if ha[i] != hb[i]), None)
        fa, fb = float.fromhex(ha[-1]), float.fromhex(hb[-1])
        out["runs"][key] = {
            "config_hash_equal": ra["config_hash"] == rb["config_hash"],
            "seeds_equal": ra["seeds"] == rb["seeds"],
            "status": [ra["status"], rb["status"]],
            "steps_equal": ra["steps_completed"] == rb["steps_completed"],
            "epoch0_bit_identical": bool(ha) and bool(hb) and ha[0] == hb[0],
            "first_differing_epoch": first,
            "bit_identical": first is None and len(ha) == len(hb),
            "final_train_mse": [fa, fb],
        }
    rs = [v for v in out["runs"].values() if "missing" not in v]
    va = {tuple(sorted((x["versions"] or {}).items())) for x in a["runs"].values() if x}
    vb = {tuple(sorted((x["versions"] or {}).items())) for x in b["runs"].values() if x}
    out["versions"] = [[dict(v) for v in va], [dict(v) for v in vb]]
    out["summary"] = {
        "runs_compared": len(rs),
        "inputs_identical": all(x["config_hash_equal"] and x["seeds_equal"] for x in rs) and out["dataset_sha256_equal"],
        "bit_identical_runs": sum(x["bit_identical"] for x in rs),
        "epoch0_identical_runs": sum(x["epoch0_bit_identical"] for x in rs),
        "earliest_divergence_epoch": min((x["first_differing_epoch"] for x in rs
                                          if x["first_differing_epoch"] is not None), default=None),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fingerprint or compare A1 experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--config", type=Path, default=REPO / "configs" / "a1" / "pilot.json")
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "make":
        fp = make(args.config)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fp, indent=1), encoding="utf-8")
        n = sum(v is not None for v in fp["runs"].values())
        print(f"fingerprint of '{fp['experiment']}': {n} runs -> {args.out}")
        return 0
    r = compare(json.loads(args.a.read_text(encoding="utf-8")), json.loads(args.b.read_text(encoding="utf-8")))
    s = r["summary"]
    print(f"inputs identical (dataset, config hashes, seeds): {s['inputs_identical']}")
    print(f"source files differing: {r['source_files_differing'] or 'none'}")
    print(f"bit-identical runs: {s['bit_identical_runs']}/{s['runs_compared']}; "
          f"identical at epoch 0: {s['epoch0_identical_runs']}; earliest divergence: epoch {s['earliest_divergence_epoch']}")
    for k, v in r["runs"].items():
        if "missing" in v or not v["bit_identical"]:
            print(f"  {k:<14} {v}")
    print(f"versions: A {r['versions'][0]}\n          B {r['versions'][1]}")
    if args.out:
        args.out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
