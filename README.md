# Hybrid EM simulations

Code for the main and appendix simulation figures in *Is $\sqrt{d}$ Separation Necessary for Gradient EM to Learn Gaussian Mixtures in High Dimensions?*

The experiments fit a two-component Gaussian mixture using ten fitted components, an EM update for the weights, and a gradient EM update for the means. The release reproduces the fixed-separation comparison at dimensions 10 and 500 and the mean/weight trajectories at dimension 1,000,000.

## Setup

The reference environment uses Python 3.14.6, NumPy 2.5.3, and Matplotlib 3.11.2 on macOS. The runner also supports Linux. A CPU is sufficient.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce all figures

```sh
make reproduce PYTHON=python
```

This runs the two configurations, replays the high-dimensional runs to record every mean and weight update, and generates four vector PDFs in `artifacts/`. It also saves numerical histories, checkpoints, and provenance in `results/`. Existing completed runs are reused when the configuration and numerical-code hashes match. No downloaded data or precomputed results are required.

| Generated figure | Content | Filename used in the manuscript |
|---|---|---|
| `artifacts/main_trajectories.pdf` | Dimensions 10 and 500, seed 0 | `em_main_trajectories.pdf` |
| `artifacts/original_trajectories.pdf` | Dimensions 10 and 500, all three seeds | `em_appendix_original_trajectories.pdf` |
| `artifacts/d1e6_parameter_trajectories.pdf` | Dimension 1,000,000, mean trajectories | `em_appendix_d1e6_mean_trajectories.pdf` |
| `artifacts/d1e6_weight_updates.pdf` | Dimension 1,000,000, fitted-weight trajectories | `em_appendix_d1e6_weight_trajectories.pdf` |

To redraw the figures after a completed run:

```sh
make figures PYTHON=python
```

The equivalent commands without Make are:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -m experiments.run --config configs/fixed_separation.json
python -m experiments.run --config configs/rebuttal_d1e6.json
python -m experiments.diagnose
python -m experiments.summarize
```

## Experiment settings

| Parameter | Fixed separation | Dimension-dependent separation |
|---|---|---|
| Configuration | `configs/fixed_separation.json` | `configs/rebuttal_d1e6.json` |
| Dimension | 10, 500 | 1,000,000 |
| True / fitted components | 2 / 10 | 2 / 10 |
| True weights | (0.3, 0.7) | (0.3, 0.7) |
| True centers | +3 and -3 on the first coordinate | +Delta/2 and -Delta/2 on the first coordinate |
| Separation Delta | 6 | `d^0.45` or `10 sqrt(d)` |
| Learning rate | 0.05 | 0.05 |
| Batch size | 8192 | 16 |
| Updates | 2000 | 500 |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Plotted states | Every 50 updates | Every update |

All covariances are identity matrices. Each fitted mean is initialized independently by drawing `Z_i ~ Categorical(0.3, 0.7)` and setting `mu_i = mu_true[Z_i] + N(0, I_d)`. All fitted weights start at 0.1. Initialization and training use full ambient-dimensional vectors in double precision.

Both updates evaluate responsibilities at the pre-update parameters on a fresh batch:

```text
pi_next[i] = mean(psi_i(X))
mu_next[i] = mu[i] + 0.05 * mean(psi_i(X) * (X - mu[i]))
```

Weights are stored in log space without floors, pruning, or restarts. The selected configurations use the direct empirical update. The mean update is not divided by the fitted weight.

Each seed determines separate initialization, training, and evaluation streams using `SeedSequence(seed).spawn(3)`. Different seeds change all stochastic parts of a run. The same seed values are reused across settings: the two high-dimensional separation settings share their initial source labels, standard Gaussian draws, and batch random draws within each seed. They therefore form three paired comparisons. Component indices retain initialization order; no component is given special treatment.

## Outputs and checks

All three seeds are shown in the appendix figures. In the recorded runs, all three seeds fit both true components at dimension 10; all three underfit at dimension 500. At dimension 1,000,000, the smaller separation leaves one active component and the larger separation leaves two. These are finite-run observations, not assertions about every seed or verification of the theorem's asymptotic conditions.

The main figure uses seed 0. The shared plotting style uses thin colored curves, gray initialization crosses, colored endpoints, and red stars for true means. The two high-dimensional plots retain the same component colors.

The replay checks final full-dimensional means and log weights against the original checkpoints bit for bit. The plot generator checks run completeness and provenance. `artifacts/figure_manifest.json` maps all twelve runs to the four figures. Held-out measurements are retained in the raw logs; the paper figures show only the specified trajectories. Evaluation uses separate random draws and does not affect training.

The reference runs used about 470 seconds for training and evaluation, plus 396 seconds for detailed replay, with a maximum process RSS of approximately 920 MiB. Checkpoints occupy approximately 461 MB. Times exclude file output, plotting, and setup and depend on hardware. Numerical-library threads are limited to one by the Makefile. Floating-point differences across platforms and library builds can affect exact numerical agreement.

`experiments/core.py` implements initialization and updates; `experiments/run.py` generates the runs; `experiments/diagnose.py` records per-update traces; and the remaining modules generate the figures and summaries.
