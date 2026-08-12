# Convenience entry points for SAR-DSL development.

BUILD_DIR ?= build
JOBS ?= $(shell nproc)

.PHONY: all llvm scalehls build dev test test-lit test-python examples bench clean

all: build

## One-time toolchain builds -------------------------------------------------

llvm:                     ## Build the in-tree LLVM/MLIR/Clang toolchain
	bash scripts/build-llvm.sh

scalehls:                 ## Build the ScaleHLS-HIDA toolchain (optional)
	bash scripts/build-scalehls.sh

## Project -------------------------------------------------------------------

build:                    ## Configure + build sar-opt, runtime, tests
	cmake -G Ninja -S . -B $(BUILD_DIR)
	ninja -C $(BUILD_DIR) -j $(JOBS)

dev: build                ## Development setup (backend symlinks)
	bash scripts/setup-dev.sh

test: test-lit test-python  ## Run everything

test-lit:                 ## MLIR FileCheck tests
	ninja -C $(BUILD_DIR) check-sar-lit

test-python:              ## Python frontend + backend tests
	PYTHONPATH=python python3 -m pytest test/python -q

examples:                 ## Focus a synthetic scene end-to-end
	PYTHONPATH=python python3 examples/wka/run_synthetic.py --n 512

bench:                    ## WKA performance benchmark
	PYTHONPATH=python python3 benchmarks/bench_wka.py --sizes 1024 4096 --numpy

clean:
	rm -rf $(BUILD_DIR)
