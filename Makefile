SHELL := /bin/bash
ROOT := $(CURDIR)
export UV_CACHE_DIR := $(ROOT)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(ROOT)/.tools/python
export GOCACHE := $(ROOT)/.cache/go-build
export GOPATH := $(ROOT)/.cache/go
export GOENV := off
export GOTOOLCHAIN := local
export GOFLAGS := -count=1
export PYTHONNOUSERSITE := 1
export PYTHONDONTWRITEBYTECODE := 1
export PATH := $(ROOT)/.tools/go/bin:$(ROOT)/.venv/bin:$(PATH)
# Offline runs never see the shared RTX 4090; CRUCIBLE_GPU_TESTS=1 opts in.
ifneq ($(CRUCIBLE_GPU_TESTS),1)
export CUDA_VISIBLE_DEVICES :=
endif

PY := $(ROOT)/.venv/bin/python
PY_ML := $(ROOT)/.venv-ml/bin/python
GO := $(ROOT)/.tools/go/bin/go
# No cache plugin: results never depend on a previous run's .pytest_cache.
PYTEST := -m pytest -q -p no:cacheprovider
# Tests that need torch; they skip under .venv and run under .venv-ml.
ML_TESTS := kernels rl scripts/tests/test_build_report_pdf.py scripts/tests/test_build_report_html.py

.PHONY: bootstrap bootstrap-python bootstrap-ml toolchains test test-python test-ml test-go \
	check lint vet replay-bin snowflake-bins
bootstrap: bootstrap-python
	bash scripts/bootstrap.sh

bootstrap-python:
	bash scripts/bootstrap_python.sh

bootstrap-ml: bootstrap
	bash scripts/bootstrap_ml.sh

# Fail fast if an interpreter or Go resolves outside the project.
toolchains:
	@for py in $(PY) $(PY_ML); do \
	  real=$$(readlink -f $$py); \
	  case $$real in $(ROOT)/.tools/python/*) ;; \
	    *) echo "$$py -> $$real is outside .tools/python; run 'make bootstrap-python'" >&2; exit 1;; \
	  esac; \
	done
	@test -x $(GO) || { echo "missing $(GO); run 'make bootstrap'" >&2; exit 1; }

# Offline suites only. GPU tests need CRUCIBLE_GPU_TESTS=1 (run them through
# scripts/with_gpu.sh); live/cloud integration tests need CRUCIBLE_LIVE_TESTS=1.
test: test-python test-ml test-go

# basetemp stays inside the project (disk share; some RL tests require an owned
# project runtime root). pytest wipes it at the start of each run.
test-python: toolchains
	@mkdir -p $(ROOT)/.cache/pytest
	$(PY) $(PYTEST) --basetemp=$(ROOT)/.cache/pytest/python

test-ml: toolchains
	@mkdir -p $(ROOT)/.cache/pytest
	$(PY_ML) $(PYTEST) --basetemp=$(ROOT)/.cache/pytest/ml $(ML_TESTS)

test-go: toolchains
	$(GO) test ./serving/...

# Router, autoscaler and replayer for the Task 14 local experiment.
t14-bins: toolchains
	$(GO) build -o $(ROOT)/.cache/t14/bin/router ./serving/router/cmd/router
	$(GO) build -o $(ROOT)/.cache/t14/bin/autoscaler ./serving/autoscaler
	$(GO) build -o $(ROOT)/.cache/t14/bin/replay ./serving/replay/cmd/replay

# Static router/autoscaler for the kind deployment (mounted into the node, run in
# the preloaded kube-proxy image); see serving/k8s/manifests.py.
k8s-bins: toolchains
	@mkdir -p $(ROOT)/.cache/k8s/bin $(ROOT)/.cache/k8s/logs
	CGO_ENABLED=0 $(GO) build -o $(ROOT)/.cache/k8s/bin/router ./serving/router/cmd/router
	CGO_ENABLED=0 $(GO) build -o $(ROOT)/.cache/k8s/bin/autoscaler ./serving/autoscaler

# Static router/autoscaler/replayer for Snowflake ML Jobs (shipped with the payload, see
# scripts/snowflake/payloads/e3_autoscale.py).
snowflake-bins: toolchains
	@mkdir -p $(ROOT)/.cache/snowflake/bin
	CGO_ENABLED=0 $(GO) build -trimpath -o $(ROOT)/.cache/snowflake/bin/router ./serving/router/cmd/router
	CGO_ENABLED=0 $(GO) build -trimpath -o $(ROOT)/.cache/snowflake/bin/autoscaler ./serving/autoscaler
	CGO_ENABLED=0 $(GO) build -trimpath -o $(ROOT)/.cache/snowflake/bin/replay ./serving/replay/cmd/replay

replay-bin: toolchains
	$(GO) build -o $(ROOT)/.cache/bin/replay ./serving/replay/cmd/replay

check: lint vet

lint: toolchains
	$(PY) -m ruff check .
	@out=$$($(ROOT)/.tools/go/bin/gofmt -l serving); \
	  if [[ -n $$out ]]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

vet: toolchains
	$(GO) vet ./serving/...

.PHONY: azure-reaper azure-down azure-down-apply resource-preflight test-resource-guards
azure-reaper:
	bash scripts/azure_reaper.sh --plan

azure-down:
	bash scripts/azure_down.sh --plan

azure-down-apply:
	bash scripts/azure_down.sh --apply

resource-preflight:
	$(PY) scripts/resource_preflight.py $(PREFLIGHT_ARGS)

test-resource-guards: toolchains
	$(PY) $(PYTEST) scripts/tests/test_azure_guards.py scripts/tests/test_gpu_guard.py \
	  scripts/tests/test_resource_preflight.py
