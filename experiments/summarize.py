"""Validate the fixed-separation and d=10^6 runs and generate all paper figures."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

# Keep font/cache writes inside the project, including in restricted environments.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
from .figures import generate_figures

METRICS = ["kl", "kl_mc_se", "max_weight", "heavy_representative_weight"]


def load_suite(path):
    manifest = json.loads((path / "manifest.json").read_text())
    runs = []
    for item in manifest["expected_runs"]:
        p = path / (item["id"] + ".json")
        if not p.exists():
            raise ValueError(f"Incomplete suite: {p} is missing. No paper artifacts generated.")
        r = json.loads(p.read_text())
        if r["config"] != item["config"] or r["code_sha256"] != manifest["code_sha256"]:
            raise ValueError(f"Provenance mismatch in {p}")
        if r["history"][-1]["step"] != r["config"]["steps"]:
            raise ValueError(f"Unfinished run: {p}")
        r["suite"] = manifest["suite"]
        runs.append(r)
    return manifest, runs


def group_runs(runs):
    groups = defaultdict(list)
    for r in runs:
        c = r["config"]
        # Full configuration except seed; never combine different estimators or batches.
        key = json.dumps({k: v for k, v in c.items() if k != "seed"}, sort_keys=True)
        groups[(r["suite"], key)].append(r)
    return groups


def summarize(runs):
    rows = []
    for (suite, key), group in group_runs(runs).items():
        config = json.loads(key)
        if len({r["config"]["seed"] for r in group}) != len(group):
            raise ValueError("Duplicate seed in group")
        row = {"suite": suite, **config, "n_seeds": len(group),
               "seeds": sorted(r["config"]["seed"] for r in group),
               "separation": group[0]["separation"],
               "collapse_count": sum(r["history"][-1]["max_weight"] >= .99 for r in group),
               "kl_below_01_count": sum(r["history"][-1]["kl"] < .1 for r in group),
               "covered_count": sum(r["initialization_covers_all_sources"] for r in group),
               "elapsed_seconds": sum(r["elapsed_seconds"] for r in group)}
        for metric in METRICS:
            values = np.array([r["history"][-1][metric] for r in group])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return sorted(rows, key=lambda r: (r["suite"], r["name"], r["dimension"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--diagnostics", type=Path, help="Default: RESULTS/diagnostics_d1e6")
    args = parser.parse_args()
    manifests, runs = [], []
    for suite in ["fixed_separation", "rebuttal_scale_d1e6"]:
        manifest, group = load_suite(args.results / suite)
        manifests.append(manifest)
        runs.extend(group)
    if len({m["code_sha256"] for m in manifests}) != 1:
        raise ValueError("Suites use different simulation code")
    rows = summarize(runs)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = []
    for r in runs:
        final.append({"suite": r["suite"], "run_id": r["run_id"], "name": r["config"]["name"],
                      "dimension": r["config"]["dimension"], "seed": r["config"]["seed"],
                      "covered": r["initialization_covers_all_sources"],
                      "elapsed_seconds": r["elapsed_seconds"],
                      **{k: r["history"][-1][k] for k in METRICS}})
    with (out / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)
    figures = generate_figures(runs, out, args.diagnostics or args.results / "diagnostics_d1e6")
    provenance = {"scope": "fixed_separation_d10_d500_and_d1e6", "parameter_policy": "fixed_separation_dimensions_10_500_other_original_parameters_retained",
                  "reproduction_status": "all_paper_figures_generated_by_supplied_code",
                  "total_runs": len(runs), "sum_run_seconds": sum(r["elapsed_seconds"] for r in runs),
                  "max_process_peak_rss_mib": max(r["process_peak_rss_mib"] for r in runs),
                  "simulation_code_sha256": manifests[0]["code_sha256"],
                  "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "figure_code_sha256": hashlib.sha256((ROOT / "experiments/figures.py").read_bytes()).hexdigest(),
                  "plot_style_sha256": hashlib.sha256((ROOT / "experiments/plot_style.py").read_bytes()).hexdigest(),
                  "figure_count": len(figures),
                  "manifest_paths": [os.path.relpath(args.results / m["suite"] / "manifest.json", ROOT) for m in manifests],
                  "machine": manifests[0]["machine"]}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "compute.tex").write_text(
        f"{len(runs)} retained reconstruction runs; {provenance['sum_run_seconds']:.1f} summed run-seconds "
        f"({provenance['sum_run_seconds']/3600:.3f} CPU-worker hours); "
        f"{provenance['max_process_peak_rss_mib']:.1f} MiB maximum process RSS.\n")
    print(f"Validated {len(runs)} completed runs; generated one main and three appendix figures in {out}")


if __name__ == "__main__":
    main()
