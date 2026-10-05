"""Run an experiment suite defined by a JSON configuration file."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
from .core import initialize, draw_batch, update, evaluate

ROOT = Path(__file__).resolve().parents[1]


def code_digest():
    h = hashlib.sha256()
    for p in [ROOT / "experiments" / "core.py", ROOT / "experiments" / "run.py"]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def cases(config):
    for base in config["experiments"]:
        grid = base.get("grid", {})
        for values in itertools.product(*grid.values()):
            case = {k: v for k, v in base.items() if k != "grid"}
            case.update(zip(grid.keys(), values))
            for seed in config["seeds"]:
                yield {**config.get("defaults", {}), **case, "seed": seed}


def validate(c):
    for key in ["dimension", "components", "batch_size", "steps", "eval_every", "eval_samples"]:
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["eval_samples"] < 2:
        raise ValueError("Need at least two held-out samples per true component")
    if c["estimator"] not in {"direct", "stein"}:
        raise ValueError("Unknown estimator")
    if c["representation"] not in {"dense", "compact"}:
        raise ValueError("Unknown representation")
    if c["sampling"] not in {"mixture", "stratified"}:
        raise ValueError("Unknown sampling scheme")
    if c["representation"] == "compact" and c["estimator"] != "stein":
        raise ValueError("A fixed compact span is valid for Stein updates only")
    if not 0 < c["learning_rate"] <= 1:
        raise ValueError("learning_rate must be in (0,1]")
    if c["separation_rule"] not in {"fixed", "power", "sqrt"}:
        raise ValueError("Unknown separation rule")


def separation(c):
    if c["separation_rule"] == "fixed":
        return float(c["separation_value"])
    if c["separation_rule"] == "power":
        return float(c["dimension"] ** c["separation_value"])
    return float(c["separation_value"] * np.sqrt(c["dimension"]))


def machine():
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            cpu = f"{platform.machine()} (CPU model unavailable)"
    return {"system": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__, "gpu": "none",
            "thread_environment": {k: os.environ.get(k) for k in
                ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}


def run_case(c):
    validate(c)
    delta = separation(c)
    streams = np.random.SeedSequence(c["seed"]).spawn(3)
    init_rng, train_rng, eval_rng = [np.random.default_rng(s) for s in streams]
    state = initialize(init_rng, c["dimension"], c["components"], c["weights"], delta,
                       c.get("geometry", "simplex"), c["representation"] == "compact")
    counts = np.bincount(state.labels, minlength=len(state.true_weights))
    orth = np.linalg.norm(state.means[:, state.signal_dimension:], axis=1)
    heavy = int(np.argmax(state.true_weights))
    heavy_indices = np.flatnonzero(state.labels == heavy)
    representative = int(heavy_indices[np.argmin(orth[heavy_indices])]) if len(heavy_indices) else None
    start = time.perf_counter()
    history = []
    grad = None
    for step in range(c["steps"] + 1):
        if step % c["eval_every"] == 0 or step == c["steps"]:
            row = evaluate(state, eval_rng, c["eval_samples"], delta, grad)
            row["step"] = step
            row["heavy_representative_weight"] = None if representative is None else row["weights"][representative]
            history.append(row)
        if step == c["steps"]:
            break
        x, labels, mass = draw_batch(train_rng, state, c["batch_size"], c["sampling"])
        grad = update(state, x, labels, mass, c["learning_rate"], c["estimator"])
    elapsed = time.perf_counter() - start
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_mb = maxrss / (1024**2 if platform.system() == "Darwin" else 1024)
    result = {
        "schema_version": 1, "config": c, "separation": delta,
        "delta_over_sqrt_d": delta / np.sqrt(c["dimension"]),
        "coordinate_dimension": state.means.shape[1],
        "initial_source_labels": state.labels.tolist(), "initial_source_counts": counts.tolist(),
        "initialization_covers_all_sources": bool(np.all(counts > 0)),
        "heavy_representative_index": representative,
        "elapsed_seconds": elapsed, "process_peak_rss_mib": maxrss_mb,
        "rss_note": "Process lifetime high-water mark, not per-case allocation",
        "history": history,
    }
    return result, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Run first N cases only; still records the full suite manifest")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    planned = list(cases(config))
    for c in planned:
        validate(c)
    out = args.output or ROOT / "results" / config["name"]
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = code_digest()
    expected = [{"id": hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12], "config": c}
                for c in planned]
    manifest = {"suite": config["name"], "config": config, "expected_runs": expected,
                "code_sha256": fingerprint, "machine": machine()}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["code_sha256"] != fingerprint or old["config"] != config:
            raise SystemExit("Output contains results from different code/config. Use a new --output directory.")
    else:
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    selected = expected if args.limit is None else expected[:args.limit]
    for item in selected:
        path = out / (item["id"] + ".json")
        if path.exists():
            print(f"Existing {path.name}", flush=True)
            continue
        result, state = run_case(item["config"])
        result["code_sha256"] = fingerprint
        result["run_id"] = item["id"]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(path.with_suffix(".npz"), means=state.means, log_weights=state.log_weights,
                            truth=state.truth, true_weights=state.true_weights, labels=state.labels)
        temporary.replace(path)
        row = result["history"][-1]
        print(f"{config['name']} {item['config']['name']} d={item['config']['dimension']} "
              f"seed={item['config']['seed']} KL={row['kl']:.5g} max_pi={row['max_weight']:.4f} "
              f"{result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
