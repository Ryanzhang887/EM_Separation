"""Replay the six d=10^6 runs, logging every mean and weight update.

Training calls the unchanged core.update. Extra measurements do not consume
the training random stream. Full final arrays and saved checkpoints must
match the original run before a trace is accepted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .core import initialize, draw_batch, log_responsibilities, update
from .run import ROOT, code_digest, machine


def stationary_coordinate_variance(probability, learning_rate, batch_size):
    """Hard, correct assignments; fresh iid mixture batches; constant step.

    e_next = (1-eta*N/B)*e + eta/B*sum_{b:Z=j} noise_b.
    E[(1-eta*N/B)^2] uses N~Binomial(B,p), including random batch counts.
    """
    p = np.asarray(probability, dtype=float)
    eta, b = learning_rate, batch_size
    return eta / (b * (2 - eta * (p + (1 - p) / b)))


def loss_parts(parallel, orthogonal, log_weights, assignment, true_weights, separation):
    """Separated-component KL approximation for a specified source mapping.

    The log-mass term is a categorical KL only for a one-to-one mapping.
    With both sources mapped to one Gaussian it includes negative source
    entropy. Omitted true/fitted mixture-overlap corrections are negligible
    only once each source overwhelmingly uses its selected component.
    """
    mu = np.asarray(parallel)[..., :, 0]
    orth = np.asarray(orthogonal)
    logpi = np.asarray(log_weights)
    assignment = np.asarray(assignment, dtype=int)
    p = np.asarray(true_weights)
    truth = np.array([separation / 2, -separation / 2])
    orth_term = .5 * np.sum(p * orth[..., assignment] ** 2, axis=-1)
    signal_term = .5 * np.sum(p * (mu[..., assignment] - truth) ** 2, axis=-1)
    mass_term = np.sum(p * (np.log(p) - logpi[..., assignment]), axis=-1)
    return orth_term, signal_term, mass_term


def state_row(state, step):
    orth = state.means[:, state.signal_dimension:]
    return {"step": step,
            "parallel_means": state.means[:, :state.signal_dimension].tolist(),
            "orthogonal_norms": np.sqrt(np.einsum("ij,ij->i", orth, orth)).tolist(),
            "weights": np.exp(state.log_weights).tolist(),
            "log_weights": state.log_weights.tolist()}


def replay(source_path):
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    c = source["config"]
    if source["code_sha256"] != code_digest():
        raise ValueError("Simulation code differs from the original run")
    if c["representation"] != "dense" or c["estimator"] != "direct":
        raise ValueError("This diagnostic is for the existing dense direct runs")
    streams = np.random.SeedSequence(c["seed"]).spawn(3)
    init_rng, train_rng = [np.random.default_rng(s) for s in streams[:2]]
    state = initialize(init_rng, c["dimension"], c["components"], c["weights"],
                       source["separation"], c["geometry"], compact=False)
    assignment = np.argmax(source["history"][-1]["conditional_responsibilities"], axis=1)
    checkpoints = {h["step"]: h for h in source["history"]}
    trace, max_checkpoint_error, max_update_error = [], 0., 0.
    start = time.perf_counter()
    for step in range(c["steps"] + 1):
        row = state_row(state, step)
        if step:
            row.update({"batch_source_counts": counts.tolist(),
                        "batch_source_to_fitted_mass": contributions.tolist(),
                        "weight_update_error": weight_error,
                        "final_mapping_batch_prediction": predicted.tolist(),
                        "final_mapping_batch_error": float(np.max(np.abs(predicted - np.exp(state.log_weights))))})
        trace.append(row)
        if step in checkpoints:
            old = checkpoints[step]
            for metric in ["parallel_means", "orthogonal_norms", "log_weights", "weights"]:
                observed, saved = np.asarray(row[metric]), np.asarray(old[metric])
                max_checkpoint_error = max(max_checkpoint_error, float(np.max(np.abs(observed - saved))))
                np.testing.assert_allclose(observed, saved, rtol=2e-12, atol=1e-10,
                                           err_msg=f"Checkpoint mismatch: step {step}, {metric}")
        if step == c["steps"]:
            break
        x, labels, mass = draw_batch(train_rng, state, c["batch_size"], c["sampling"])
        # Read-only responsibility measurement at the same PRE-update state.
        responsibilities = np.exp(log_responsibilities(x, state.means, state.log_weights))
        counts = np.bincount(labels, minlength=len(state.true_weights))
        contributions = np.array([(mass[labels == j, None] * responsibilities[labels == j]).sum(axis=0)
                                  for j in range(len(state.true_weights))])
        update(state, x, labels, mass, c["learning_rate"], c["estimator"])
        weight_error = float(np.max(np.abs(contributions.sum(axis=0) - np.exp(state.log_weights))))
        max_update_error = max(max_update_error, weight_error)
        # Huge log scores lose a few ulps during softmax subtraction; the core
        # explicitly renormalizes the new weights. Compare normalized masses.
        normalized_error = float(np.max(np.abs(contributions.sum(axis=0) / contributions.sum()
                                               - np.exp(state.log_weights))))
        if normalized_error > 1e-12 or weight_error > 1e-7:
            raise ValueError(f"Weight update mismatch at step {step+1}: "
                             f"raw={weight_error}, normalized={normalized_error}")
        predicted = np.zeros(c["components"])
        np.add.at(predicted, assignment, counts / c["batch_size"])
    elapsed = time.perf_counter() - start
    with np.load(source_path.with_suffix(".npz")) as checkpoint:
        for key in ["means", "log_weights", "truth", "true_weights", "labels"]:
            if not np.array_equal(getattr(state, key), checkpoint[key]):
                raise ValueError(f"Final full array differs from original checkpoint: {key}")
    errors = np.array([r["final_mapping_batch_error"] for r in trace[1:]])
    remaining_max = np.maximum.accumulate(errors[::-1])[::-1]
    stable = np.flatnonzero(remaining_max < 1e-8)
    stable_step = int(stable[0] + 1) if len(stable) else None
    final = trace[-1]
    parts = [float(v) for v in loss_parts(final["parallel_means"], final["orthogonal_norms"],
                                         final["log_weights"], assignment, c["weights"], source["separation"])]
    p = np.asarray(c["weights"])
    v = stationary_coordinate_variance(p, c["learning_rate"], c["batch_size"])
    return {"source_run_id": source["run_id"], "source_suite": "rebuttal_scale_d1e6",
            "source_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "simulation_code_sha256": source["code_sha256"],
            "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config": c, "separation": source["separation"],
            "initial_source_labels": source["initial_source_labels"],
            "final_assignment_zero_based": assignment.tolist(),
            "first_persistent_batch_assignment_step": stable_step,
            "assignment_tolerance": 1e-8,
            "max_weight_update_error": max_update_error,
            "max_checkpoint_absolute_error": max_checkpoint_error,
            "final_arrays_bitwise_equal": True, "elapsed_seconds": elapsed,
            "endpoint_loss_parts": dict(zip(["orthogonal", "signal", "log_mass"], parts)),
            "endpoint_separated_kl_approximation": sum(parts),
            "endpoint_heldout_kl": source["history"][-1]["kl"],
            "endpoint_heldout_kl_mc_se": source["history"][-1]["kl_mc_se"],
            "hard_assignment_noise_model": {
                "scope": "One fitted Gaussian assigned correctly to each true source",
                "stationary_coordinate_variance": v.tolist(),
                "stationary_orthogonal_rms": np.sqrt((c["dimension"]-1)*v).tolist(),
                "stationary_orthogonal_kl": float(.5*(c["dimension"]-1)*np.dot(p,v))},
            "history": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/rebuttal_scale_d1e6")
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics_d1e6")
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    if manifest["suite"] != "rebuttal_scale_d1e6":
        raise ValueError("Expected the existing d=10^6 suite")
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    code_dir = args.output / "code"
    code_dir.mkdir(exist_ok=True)
    (code_dir / (fingerprint + ".py")).write_bytes(Path(__file__).read_bytes())
    total_seconds = 0.
    trace_fingerprints = set()
    for item in manifest["expected_runs"]:
        target = args.output / (item["id"] + ".json")
        source_path = args.source / (item["id"] + ".json")
        if target.exists():
            result = json.loads(target.read_text())
            old_code = code_dir / (result["diagnostic_code_sha256"] + ".py")
            if (not old_code.exists() or hashlib.sha256(old_code.read_bytes()).hexdigest() != result["diagnostic_code_sha256"] or
                result["source_json_sha256"] != hashlib.sha256(source_path.read_bytes()).hexdigest()):
                raise ValueError("Existing diagnostics lack matching source/code provenance")
        else:
            result = replay(source_path)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            temporary.replace(target)
        total_seconds += result["elapsed_seconds"]
        trace_fingerprints.add(result["diagnostic_code_sha256"])
        c = result["config"]
        print(f"{c['name']} seed={c['seed']}: exact final replay; "
              f"batch assignment stable from update {result['first_persistent_batch_assignment_step']}; "
              f"weight-update residual {result['max_weight_update_error']:.2g}; "
              f"{result['elapsed_seconds']:.1f}s", flush=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "source_suite": manifest["suite"], "source_runs": manifest["expected_runs"],
        "simulation_code_sha256": manifest["code_sha256"],
        "diagnostic_code_sha256": fingerprint, "machine": machine(),
        "trace_code_sha256": sorted(trace_fingerprints),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sum_replay_seconds": total_seconds,
        "scope": "Exact replays for per-update measurements; no new experiment settings or seeds"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
