PYTHON ?= python3
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

.PHONY: all reproduce figures

all: reproduce

reproduce:
	$(PYTHON) -m experiments.run --config configs/fixed_separation.json
	$(PYTHON) -m experiments.run --config configs/high_dimension.json
	$(PYTHON) -m experiments.diagnose
	$(MAKE) figures

figures:
	$(PYTHON) -m experiments.summarize
