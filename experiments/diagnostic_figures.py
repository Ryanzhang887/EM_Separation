"""Report only mean and weight trajectories for the existing d=10^6 runs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plot_style import WIDTH, TAB10, LABELS, component_style, zero_line


def arrays(trace):
    h = trace["history"]
    return (np.array([r["step"] for r in h]),
            np.array([r["parallel_means"] for r in h]),
            np.array([r["orthogonal_norms"] for r in h]),
            np.array([r["weights"] for r in h]),
            np.array([r["log_weights"] for r in h]))


def save(fig, out, filename):
    fig.savefig(out / filename, bbox_inches="tight", metadata={"Creator": "experiments.summarize diagnostics"})
    plt.close(fig)


def component_legend(fig):
    handles = [Line2D([], [], color=TAB10[i], linewidth=.9, label=rf"$i={i+1}$") for i in range(10)]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(.5, 1), fontsize=7)


def trajectories(traces, out):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.5))
    for r in traces:
        c = r["config"]
        ax = axes[c["seed"], 0 if c["name"] == "power_045" else 1]
        _, parallel, orth, _, _ = arrays(r)
        for i in range(c["components"]):
            style = component_style(i)
            ax.plot(parallel[:, i, 0], orth[:, i], **style)
            ax.plot(parallel[0, i, 0], orth[0, i], marker="x", color=".6", linestyle="none")
            ax.plot(parallel[-1, i, 0], orth[-1, i], marker="o", color=style["color"], linestyle="none")
        delta = r["separation"]
        ax.scatter([delta / 2, -delta / 2], [0, 0], color="red", marker="*", s=45, zorder=5)
        for i in sorted(set(r["final_assignment_zero_based"])):
            ax.annotate(rf"$i={i+1}$", (parallel[-1, i, 0], orth[-1, i]), xytext=(3, 6),
                        textcoords="offset points", fontsize=7, color=TAB10[i])
        zero_line(ax)
        ax.set_ylim(-45, 1100)
        ax.set_xlim(-.65 * delta, .65 * delta)
        ax.set_title(f"Seed {c['seed']}, " + LABELS[c["name"]], loc="left")
        ax.set_xlabel(r"$\mu_{i,1}$")
        ax.set_ylabel(r"$\|\mu_{i,2:d}\|_2$")
    component_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, .91), h_pad=.9, w_pad=1.2)
    save(fig, out, "d1e6_parameter_trajectories.pdf")


def weights(traces, out):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.3), sharex=True, sharey=True)
    for r in traces:
        c = r["config"]
        ax = axes[c["seed"], 0 if c["name"] == "power_045" else 1]
        steps, _, _, pi, _ = arrays(r)
        for i in range(c["components"]):
            ax.plot(steps, pi[:, i], **component_style(i))
            ax.plot(steps[-1], pi[-1, i], color=TAB10[i], marker="o", linestyle="none")
        for p in [.3, .7]:
            ax.axhline(p, color=".55", linewidth=.6, linestyle=":", zorder=0)
        ax.set_title(f"Seed {c['seed']}, " + LABELS[c["name"]], loc="left")
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
        ax.set_ylim(-.03, 1.04)
        ax.set_yticks([0, .3, .7, 1])
        ax.set_ylabel(r"Fitted weight $\pi_i$")
        if c["seed"] == 2:
            ax.set_xlabel("Iteration")
    component_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, .91), h_pad=.9, w_pad=1.2)
    save(fig, out, "d1e6_weight_updates.pdf")


def generate_diagnostic_figures(source_runs, diagnostic_path, out):
    manifest = json.loads((diagnostic_path / "manifest.json").read_text())
    traces, rows = [], []
    code_hash = hashlib.sha256((Path(__file__).parent / "diagnose.py").read_bytes()).hexdigest()
    for source in source_runs:
        r = json.loads((diagnostic_path / (source["run_id"] + ".json")).read_text())
        archived_code = diagnostic_path / "code" / (r["diagnostic_code_sha256"] + ".py")
        if (r["config"] != source["config"] or
            r["simulation_code_sha256"] != source["code_sha256"] or
            not archived_code.exists() or hashlib.sha256(archived_code.read_bytes()).hexdigest() != r["diagnostic_code_sha256"] or
            not r["final_arrays_bitwise_equal"]):
            raise ValueError("Diagnostic provenance or replay verification mismatch")
        if len(r["history"]) != source["config"]["steps"] + 1:
            raise ValueError("Incomplete per-update diagnostic")
        traces.append(r)
        h = r["history"][-1]
        a = r["final_assignment_zero_based"]
        rows.append({"run_id": source["run_id"], "name": source["config"]["name"],
                     "seed": source["config"]["seed"],
                     "fitted_indices_for_true_sources_1_based": [i+1 for i in a],
                     "final_assigned_weights": [h["weights"][i] for i in a],
                     "final_assigned_signal_means": [h["parallel_means"][i][0] for i in a],
                     "final_assigned_orthogonal_norms": [h["orthogonal_norms"][i] for i in a],
                     "final_batch_source_counts": h["batch_source_counts"],
                     "missing_source_batch_steps": [v["step"] for v in r["history"][1:] if min(v["batch_source_counts"]) == 0],
                     "batch_mapping_discrepancy_steps": [v["step"] for v in r["history"][1:] if v["final_mapping_batch_error"] > 1e-8],
                     "first_persistent_batch_assignment_step": r["first_persistent_batch_assignment_step"],
                     **r["endpoint_loss_parts"],
                     "separated_kl_approximation": r["endpoint_separated_kl_approximation"],
                     "heldout_kl": r["endpoint_heldout_kl"],
                     "heldout_kl_mc_se": r["endpoint_heldout_kl_mc_se"],
                     "max_weight_update_error": r["max_weight_update_error"],
                     "final_arrays_bitwise_equal": r["final_arrays_bitwise_equal"]})
    if len(traces) != 6 or len(manifest["source_runs"]) != 6:
        raise ValueError("Expected six exact diagnostic replays")
    trajectories(traces, out)
    weights(traces, out)
    with (out / "diagnostic_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "diagnostic_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "diagnostic_provenance.json").write_text(json.dumps({
        "scope": "Fitted-mean and fitted-weight trajectories of the existing six d=10^6 runs",
        "replays": 6, "new_experiment_settings": 0,
        "diagnostic_code_sha256": code_hash,
        "trace_code_sha256": manifest["trace_code_sha256"],
        "diagnostic_figure_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sum_replay_seconds": manifest["sum_replay_seconds"],
        "all_final_arrays_bitwise_equal": True, "machine": manifest["machine"]}, indent=2) + "\n")
    (out / "diagnostic_compute.tex").write_text(
        f"Six exact diagnostic replays add {manifest['sum_replay_seconds']:.1f} run-seconds "
        "for per-update logging; they introduce no new settings or seeds.\n")
    entries = []
    for filename, selected, metrics in [
        ("d1e6_parameter_trajectories.pdf", traces, ["parallel_means", "orthogonal_norms"]),
        ("d1e6_weight_updates.pdf", traces, ["weights"]),
    ]:
        entries.append({"file": filename, "runs": [r["source_suite"]+"/"+r["source_run_id"] for r in selected],
                        "metrics": metrics, "origin": "User-requested mean and weight trajectories; exact replay"})
    return entries
