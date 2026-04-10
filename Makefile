SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help

PYTHON ?= $(if $(wildcard $(CURDIR)/.venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
MATURIN ?= $(if $(wildcard $(CURDIR)/.venv/bin/maturin),$(CURDIR)/.venv/bin/maturin,maturin)
PIP ?= $(PYTHON) -m pip
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache

export UV_CACHE_DIR
export PYTHONDONTWRITEBYTECODE := 1

define install_shared_lib
	rm -f "$(2)"
	install -m 755 "$(1)" "$(2)"
endef

PYTHON_BINDINGS_DIR := $(CURDIR)/bindings/python
AUTOAGENTS_DIR := $(PYTHON_BINDINGS_DIR)/autoagents
GUARDRAILS_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-guardrails
LLAMACPP_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-llamacpp
MISTRALRS_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-mistralrs
LLAMACPP_CUDA_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-llamacpp-cuda
LLAMACPP_METAL_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-llamacpp-metal
LLAMACPP_VULKAN_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-llamacpp-vulkan
MISTRALRS_CUDA_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-mistralrs-cuda
MISTRALRS_METAL_DIR := $(PYTHON_BINDINGS_DIR)/autoagents-mistralrs-metal
PYTHON_BINDINGS_PYTEST_CONFIG := $(PYTHON_BINDINGS_DIR)/pytest.ini

BASE_TARGET_DIR := $(CURDIR)/target/python-bindings/base
GUARDRAILS_TARGET_DIR := $(CURDIR)/target/python-bindings/guardrails
LLAMACPP_TARGET_DIR := $(CURDIR)/target/python-bindings/llamacpp
MISTRALRS_TARGET_DIR := $(CURDIR)/target/python-bindings/mistralrs
LLAMACPP_CUDA_TARGET_DIR := $(CURDIR)/target/python-cuda/llamacpp
MISTRALRS_CUDA_TARGET_DIR := $(CURDIR)/target/python-cuda/mistralrs
LLAMACPP_METAL_TARGET_DIR := $(CURDIR)/target/python-metal/llamacpp
MISTRALRS_METAL_TARGET_DIR := $(CURDIR)/target/python-metal/mistralrs
LLAMACPP_VULKAN_TARGET_DIR := $(CURDIR)/target/python-vulkan/llamacpp

PYTHON_BINDINGS_TEST_PATHS := \
	$(AUTOAGENTS_DIR)/tests \
	$(GUARDRAILS_DIR)/tests \
	$(LLAMACPP_DIR)/tests \
	$(MISTRALRS_DIR)/tests

PYTHON_BINDINGS_PYTHONPATH := $(AUTOAGENTS_DIR):$(GUARDRAILS_DIR):$(LLAMACPP_DIR):$(MISTRALRS_DIR)
RUST_COVERAGE_OUTPUT_DIR := $(CURDIR)/target/tarpaulin-workspace

.PHONY: \
	help \
	coverage-rust \
	python-bindings-clean \
	python-bindings-check-tools \
	python-bindings-install-test-deps \
	python-bindings-check-test-deps \
	python-bindings-build-base \
	python-bindings-build-guardrails \
	python-bindings-build-llamacpp \
	python-bindings-build-mistralrs \
	python-bindings-build-llamacpp-only \
	python-bindings-build-mistralrs-only \
	python-bindings-build-fast \
	python-bindings-build \
	python-bindings-build-cuda \
	python-bindings-build-metal \
	python-bindings-build-vulkan \
	python-bindings-test \
	python-bindings-test-clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  coverage-rust              Run the workspace-wide Rust Tarpaulin coverage report' \
		'  python-bindings-build-base        Build and install the core Python binding' \
		'  python-bindings-build-guardrails  Build and install the guardrails Python binding' \
		'  python-bindings-build-llamacpp    Build and install the llama.cpp Python binding' \
		'  python-bindings-build-mistralrs   Build and install the mistral-rs Python binding' \
		'  python-bindings-install-test-deps Install pytest, pytest-asyncio, and pytest-cov into the active Python env' \
		'  python-bindings-build-llamacpp-only Clean, build base + llama.cpp bindings' \
		'  python-bindings-build-mistralrs-only Clean, build base + mistral-rs bindings' \
		'  python-bindings-build-fast   Incrementally build and install CPU Python bindings' \
		'  python-bindings-build        Clean, build, and install CPU Python bindings' \
		'  python-bindings-build-cuda   Clean, build CPU bindings, then CUDA variants' \
		'  python-bindings-build-metal  Clean, build CPU bindings, then Metal variants' \
		'  python-bindings-build-vulkan Clean, build CPU bindings, then llama.cpp Vulkan variant' \
		'  python-bindings-test         Incrementally build CPU bindings, then run Python binding tests' \
		'  python-bindings-test-clean   Clean, build CPU bindings, then run Python binding tests'

coverage-rust:
	cargo tarpaulin \
		--engine llvm \
		--timeout 220 \
		--ignore-panics \
		--workspace \
		--features full \
		--out Xml \
		--out Html \
		--output-dir "$(RUST_COVERAGE_OUTPUT_DIR)" \
		--exclude-files "examples/*"

python-bindings-clean:
	rm -rf "$(CURDIR)/target/python-bindings" \
		"$(CURDIR)/target/python-cuda" \
		"$(CURDIR)/target/python-metal" \
		"$(CURDIR)/target/python-vulkan" \
		"$(CURDIR)/.pytest_cache"
	find "$(PYTHON_BINDINGS_DIR)" \
		-type d \( -name "__pycache__" -o -name ".pytest_cache" \) \
		-prune -exec rm -rf {} +
	find "$(PYTHON_BINDINGS_DIR)" \
		-type f \( \
			-name "*.pyc" -o \
			-name "*.pyo" -o \
			-name "autoagents_py*.so" -o \
			-name "_autoagents*.so" -o \
			-name "_autoagents*.abi3.so" \
		\) \
		-delete

python-bindings-check-tools:
	@command -v "$(MATURIN)" >/dev/null 2>&1 || { \
		echo "error: missing required command '$(MATURIN)'"; \
		exit 1; \
	}
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { \
		echo "error: missing required interpreter '$(PYTHON)'"; \
		exit 1; \
	}

python-bindings-install-test-deps: python-bindings-check-tools
	"$(PIP)" install -U pytest pytest-asyncio pytest-cov

python-bindings-check-test-deps:
	@missing=""; \
	for module in pytest pytest_cov pytest_asyncio; do \
		if ! "$(PYTHON)" -c "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)" "$$module"; then \
			if [ -n "$$missing" ]; then \
				missing="$$missing, $$module"; \
			else \
				missing="$$module"; \
			fi; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "error: missing Python test dependencies: $$missing" >&2; \
		echo "run 'make python-bindings-install-test-deps' to install them into $(PYTHON)" >&2; \
		exit 1; \
	fi

python-bindings-build-base:
	@echo "==> autoagents-py (base)"
	CARGO_TARGET_DIR="$(BASE_TARGET_DIR)" \
	"$(MATURIN)" develop --release --manifest-path "$(AUTOAGENTS_DIR)/Cargo.toml"

python-bindings-build-guardrails: python-bindings-build-base
	@echo "==> autoagents-guardrails-py"
	CARGO_TARGET_DIR="$(GUARDRAILS_TARGET_DIR)" \
	"$(MATURIN)" develop --release --manifest-path "$(GUARDRAILS_DIR)/Cargo.toml"
	$(call install_shared_lib,$(GUARDRAILS_TARGET_DIR)/maturin/lib_autoagents_guardrails.so,$(GUARDRAILS_DIR)/autoagents_guardrails/_autoagents_guardrails.abi3.so)

python-bindings-build-llamacpp: python-bindings-build-base
	@echo "==> autoagents-llamacpp-py (CPU)"
	CARGO_TARGET_DIR="$(LLAMACPP_TARGET_DIR)" \
	"$(MATURIN)" develop --release --manifest-path "$(LLAMACPP_DIR)/Cargo.toml"
	$(call install_shared_lib,$(LLAMACPP_TARGET_DIR)/maturin/lib_autoagents_llamacpp.so,$(LLAMACPP_DIR)/autoagents_llamacpp/_autoagents_llamacpp.abi3.so)

python-bindings-build-mistralrs: python-bindings-build-base
	@echo "==> autoagents-mistral-rs-py (CPU)"
	CARGO_TARGET_DIR="$(MISTRALRS_TARGET_DIR)" \
	"$(MATURIN)" develop --release --manifest-path "$(MISTRALRS_DIR)/Cargo.toml"
	$(call install_shared_lib,$(MISTRALRS_TARGET_DIR)/maturin/lib_autoagents_mistral_rs.so,$(MISTRALRS_DIR)/autoagents_mistral_rs/_autoagents_mistral_rs.abi3.so)

python-bindings-build-llamacpp-only: python-bindings-clean python-bindings-check-tools python-bindings-build-llamacpp
	@echo "ok: base + llama.cpp Python bindings installed"

python-bindings-build-mistralrs-only: python-bindings-clean python-bindings-check-tools python-bindings-build-mistralrs
	@echo "ok: base + mistral-rs Python bindings installed"

python-bindings-build-fast: python-bindings-check-tools python-bindings-build-guardrails python-bindings-build-llamacpp python-bindings-build-mistralrs
	@echo "ok: CPU Python bindings installed"

python-bindings-build: python-bindings-clean python-bindings-build-fast
	@echo "ok: CPU Python bindings installed"

python-bindings-build-cuda: python-bindings-build
	@command -v nvcc >/dev/null 2>&1 || { \
		echo "error: nvcc not found; install CUDA Toolkit and ensure it is on PATH"; \
		exit 1; \
	}
	@echo "==> autoagents-llamacpp-cuda"
	CARGO_TARGET_DIR="$(LLAMACPP_CUDA_TARGET_DIR)" \
	CMAKE_POSITION_INDEPENDENT_CODE="ON" \
	CMAKE_C_FLAGS="-fPIC" \
	CMAKE_CXX_FLAGS="-fPIC" \
	CMAKE_CUDA_FLAGS="-Xcompiler=-fPIC" \
	"$(MATURIN)" develop --release --features cuda \
		--manifest-path "$(LLAMACPP_CUDA_DIR)/Cargo.toml"
	$(call install_shared_lib,$(LLAMACPP_CUDA_TARGET_DIR)/maturin/lib_autoagents_llamacpp_cuda.so,$(LLAMACPP_CUDA_DIR)/autoagents_llamacpp_cuda/_autoagents_llamacpp_cuda.abi3.so)
	@echo "==> autoagents-mistral-rs-cuda"
	CARGO_TARGET_DIR="$(MISTRALRS_CUDA_TARGET_DIR)" \
	CMAKE_POSITION_INDEPENDENT_CODE="ON" \
	CMAKE_C_FLAGS="-fPIC" \
	CMAKE_CXX_FLAGS="-fPIC" \
	CMAKE_CUDA_FLAGS="-Xcompiler=-fPIC" \
	"$(MATURIN)" develop --release --features cuda \
		--manifest-path "$(MISTRALRS_CUDA_DIR)/Cargo.toml"
	$(call install_shared_lib,$(MISTRALRS_CUDA_TARGET_DIR)/maturin/lib_autoagents_mistral_rs_cuda.so,$(MISTRALRS_CUDA_DIR)/autoagents_mistral_rs_cuda/_autoagents_mistral_rs_cuda.abi3.so)
	@echo "ok: CPU + CUDA Python bindings installed"

python-bindings-build-metal: python-bindings-build
	@echo "==> autoagents-llamacpp-metal"
	CARGO_TARGET_DIR="$(LLAMACPP_METAL_TARGET_DIR)" \
	"$(MATURIN)" develop --release --features metal \
		--manifest-path "$(LLAMACPP_METAL_DIR)/Cargo.toml"
	$(call install_shared_lib,$(LLAMACPP_METAL_TARGET_DIR)/maturin/lib_autoagents_llamacpp_metal.so,$(LLAMACPP_METAL_DIR)/autoagents_llamacpp_metal/_autoagents_llamacpp_metal.abi3.so)
	@echo "==> autoagents-mistral-rs-metal"
	CARGO_TARGET_DIR="$(MISTRALRS_METAL_TARGET_DIR)" \
	"$(MATURIN)" develop --release --features metal \
		--manifest-path "$(MISTRALRS_METAL_DIR)/Cargo.toml"
	$(call install_shared_lib,$(MISTRALRS_METAL_TARGET_DIR)/maturin/lib_autoagents_mistral_rs_metal.so,$(MISTRALRS_METAL_DIR)/autoagents_mistral_rs_metal/_autoagents_mistral_rs_metal.abi3.so)
	@echo "ok: CPU + Metal Python bindings installed"

python-bindings-build-vulkan: python-bindings-build
	@echo "==> autoagents-llamacpp-vulkan"
	CARGO_TARGET_DIR="$(LLAMACPP_VULKAN_TARGET_DIR)" \
	"$(MATURIN)" develop --release --features vulkan \
		--manifest-path "$(LLAMACPP_VULKAN_DIR)/Cargo.toml"
	$(call install_shared_lib,$(LLAMACPP_VULKAN_TARGET_DIR)/maturin/lib_autoagents_llamacpp_vulkan.so,$(LLAMACPP_VULKAN_DIR)/autoagents_llamacpp_vulkan/_autoagents_llamacpp_vulkan.abi3.so)
	@echo "ok: CPU + Vulkan Python bindings installed"

python-bindings-test: python-bindings-build-fast python-bindings-check-test-deps
	PYTHONPATH="$(PYTHON_BINDINGS_PYTHONPATH):$${PYTHONPATH:-}" \
	"$(PYTHON)" -m pytest \
		-c "$(PYTHON_BINDINGS_PYTEST_CONFIG)" \
		$(PYTHON_BINDINGS_TEST_PATHS)

python-bindings-test-clean: python-bindings-build python-bindings-check-test-deps
	PYTHONPATH="$(PYTHON_BINDINGS_PYTHONPATH):$${PYTHONPATH:-}" \
	"$(PYTHON)" -m pytest \
		-c "$(PYTHON_BINDINGS_PYTEST_CONFIG)" \
		--cov \
		--cov-config "$(PYTHON_BINDINGS_DIR)/.coveragerc" \
		--cov-report term-missing:skip-covered \
		$(PYTHON_BINDINGS_TEST_PATHS)
