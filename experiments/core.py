"""Identity-covariance hybrid EM, with separate stochastic and Stein estimators.

All updates use responsibilities evaluated at the OLD weights and means.
Weights are stored in log space; no floors, pruning, or restarts are applied.
The Stein estimator is an unbiased estimator of the population gradient, not
the same finite-sample update as the direct estimator.
"""
from dataclasses import dataclass
import numpy as np


def logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a, dtype=np.float64)
    top = np.max(a, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        result = safe + np.log(np.sum(np.exp(a - safe), axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def simplex(m, separation):
    """Centered regular simplex in m-1 orthonormal coordinates."""
    if m < 2 or separation <= 0:
        raise ValueError("Need at least two components and positive separation")
    basis = np.zeros((m, m - 1))
    for j in range(m - 1):
        basis[:j + 1, j] = 1 / np.sqrt((j + 1) * (j + 2))
        basis[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))
    return separation / np.sqrt(2) * basis


def orthogonal_gaussians(rng, n, dimension):
    """Compact coordinates of n iid N(0,I_dimension) vectors, up to rotation.

    Bartlett factorization samples their exact Wishart Gram matrix when
    dimension >= n. For smaller dimension, draw the vectors directly.
    This preserves ambient initialization noise, rather than replacing d by n.
    """
    if dimension < 0:
        raise ValueError("Negative orthogonal dimension")
    if dimension < n:
        return rng.normal(size=(n, dimension))
    factor = np.tril(rng.normal(size=(n, n)), k=-1)
    factor[np.diag_indices(n)] = np.sqrt(rng.chisquare(dimension - np.arange(n)))
    return factor


@dataclass
class State:
    means: np.ndarray
    log_weights: np.ndarray
    truth: np.ndarray
    true_weights: np.ndarray
    labels: np.ndarray
    signal_dimension: int
    ambient_dimension: int


def initialize(rng, d, n, weights, separation, geometry="simplex", compact=False):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 2 or not np.all(weights > 0):
        raise ValueError("True weights must be a positive vector")
    if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("True weights must sum to one")
    if d < len(weights) - 1 or n < 1:
        raise ValueError("Invalid dimension or fitted component count")
    if geometry == "line":
        if len(weights) != 2:
            raise ValueError("Line geometry requires two true components")
        signal = np.array([[separation / 2], [-separation / 2]])
    elif geometry == "simplex":
        signal = simplex(len(weights), separation)
    else:
        raise ValueError("Unknown geometry")
    rank = signal.shape[1]
    labels = rng.choice(len(weights), n, p=weights)
    parallel = signal[labels] + rng.normal(size=(n, rank))
    orthogonal = (orthogonal_gaussians(rng, n, d - rank) if compact
                  else rng.normal(size=(n, d - rank)))
    means = np.concatenate([parallel, orthogonal], axis=1)
    truth = np.pad(signal, ((0, 0), (0, means.shape[1] - rank)))
    return State(means, np.full(n, -np.log(n)), truth, weights, labels, rank, d)


def draw_batch(rng, state, batch_size, sampling):
    """For stratified sampling batch_size means samples PER true component."""
    if sampling == "mixture":
        labels = rng.choice(len(state.true_weights), batch_size, p=state.true_weights)
        sample_weights = np.full(batch_size, 1 / batch_size)
    elif sampling == "stratified":
        labels = np.repeat(np.arange(len(state.true_weights)), batch_size)
        sample_weights = np.repeat(state.true_weights / batch_size, batch_size)
    else:
        raise ValueError("Unknown sampling scheme")
    x = rng.normal(size=(len(labels), state.means.shape[1]))
    x += state.truth[labels]
    return x, labels, sample_weights


def log_responsibilities(x, means, log_weights):
    scores = x @ means.T - 0.5 * np.sum(means * means, axis=1) + log_weights
    return scores - logsumexp(scores, axis=1, keepdims=True)


def update(state, x, labels, sample_weights, learning_rate, estimator):
    if not 0 < learning_rate <= 1:
        raise ValueError("Use a learning rate in (0,1]")
    log_r = log_responsibilities(x, state.means, state.log_weights)
    # Keep very small weights finite in log space, even if exp(log_r) underflows.
    new_log_weights = logsumexp(log_r + np.log(sample_weights[:, None]), axis=0)
    new_log_weights -= logsumexp(new_log_weights)
    r = np.exp(log_r)
    weighted_r = sample_weights[:, None] * r
    if estimator == "direct":
        # Gradient ascent in means: E[psi_i(X) (X-mu_i)]. No division by pi_i.
        direction = weighted_r.T @ x - weighted_r.sum(axis=0)[:, None] * state.means
    elif estimator == "stein":
        # Lemma 1: grad L_i = E[psi_i(X) sum_k psi_k(X)(mu_k-mu*_Z)].
        direction = -(weighted_r.T @ r) @ state.means + weighted_r.T @ state.truth[labels]
    else:
        raise ValueError("Unknown gradient estimator")
    state.means = state.means + learning_rate * direction
    state.log_weights = new_log_weights
    if not np.all(np.isfinite(state.means)) or not np.all(np.isfinite(new_log_weights)):
        raise FloatingPointError("Nonfinite iterate; no silent repair applied")
    return np.linalg.norm(direction, axis=1)


def gram_factor(means):
    """Factor M M^T for exact distribution of Gaussian dot products."""
    gram = means @ means.T
    gram = (gram + gram.T) / 2
    values, vectors = np.linalg.eigh(gram)
    scale = max(1.0, float(np.max(values)))
    if np.min(values) < -1e-10 * scale:
        raise FloatingPointError("Gram matrix not positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0))


def evaluate(state, rng, samples_per_component, separation, gradient_norms=None):
    """Independent, stratified Monte Carlo KL and mechanism measurements.

    Sampling joint Gaussian dot products is distributionally identical to
    sampling full-dimensional held-out observations. It does not change training.
    The common -||x||^2/2 and d log(2*pi)/2 terms cancel in log p*/p.
    The KL estimate is NOT clipped to zero. Its MC standard error is separate
    from across-seed standard deviations reported by the aggregation script.
    """
    m, n = len(state.true_weights), len(state.log_weights)
    all_means = np.concatenate([state.truth, state.means])
    factor = gram_factor(all_means)
    norm2 = np.sum(all_means * all_means, axis=1)
    means_at_truth = state.truth @ all_means.T - norm2 / 2
    component_kl, component_var, responsibilities = [], [], []
    for j in range(m):
        scores = rng.normal(size=(samples_per_component, m + n)) @ factor.T + means_at_truth[j]
        true_scores = scores[:, :m] + np.log(state.true_weights)
        fit_scores = scores[:, m:] + state.log_weights
        true_logp = logsumexp(true_scores, axis=1)
        fit_logp = logsumexp(fit_scores, axis=1)
        ratios = true_logp - fit_logp
        component_kl.append(float(np.mean(ratios)))
        component_var.append(float(np.var(ratios, ddof=1) / samples_per_component))
        responsibilities.append(np.mean(np.exp(fit_scores - fit_logp[:, None]), axis=0))
    # Subtraction is safe here: distance >= 0; a tiny negative is roundoff.
    distance2 = (np.sum(state.means**2, axis=1)[:, None]
                 + np.sum(state.truth**2, axis=1)[None, :] - 2 * state.means @ state.truth.T)
    distances = np.sqrt(np.maximum(distance2, 0))
    weights = np.exp(state.log_weights)
    nearest = np.argmin(distances, axis=1)
    assigned_mass = np.bincount(nearest, weights=weights, minlength=m)
    orthogonal_norms = np.linalg.norm(state.means[:, state.signal_dimension:], axis=1)
    coverage = np.min(distances, axis=0)
    result = {
        "kl": float(np.dot(state.true_weights, component_kl)),
        "kl_mc_se": float(np.sqrt(np.dot(state.true_weights**2, component_var))),
        "max_weight": float(np.max(weights)),
        "effective_components": float(np.exp(-np.dot(weights, state.log_weights))),
        "active_components_001": int(np.sum(weights >= 0.01)),
        "min_log_weight": float(np.min(state.log_weights)),
        "linear_weight_underflows": int(np.sum(weights == 0)),
        "max_nearest_mean_error": float(np.max(coverage)),
        "max_nearest_mean_error_over_separation": float(np.max(coverage) / separation),
        "assigned_weight_l1": float(np.sum(np.abs(assigned_mass - state.true_weights))),
        "weights": weights.tolist(), "log_weights": state.log_weights.tolist(),
        "orthogonal_norms": orthogonal_norms.tolist(),
        "parallel_means": state.means[:, :state.signal_dimension].tolist(),
        "nearest_mean_errors": coverage.tolist(),
        "conditional_responsibilities": np.asarray(responsibilities).tolist(),
        "gradient_norms": None if gradient_norms is None else gradient_norms.tolist(),
    }
    return result
