# Convenience entry points for SAR-DSL development.

BUILD_DIR ?= build
ARTIFACT_DIR ?= $(BUILD_DIR)/artifacts
JOBS ?= 16
CMAKE_ARGS ?=

.PHONY: all llvm build test test-lit test-python examples bench clean

all: build

## One-time toolchain builds -------------------------------------------------

llvm:                       ## Build the in-tree LLVM/MLIR/Clang toolchain
	bash scripts/build-llvm.sh

## Project -------------------------------------------------------------------

build:                      ## Configure + build sar-opt, runtime, tests
	cmake -G Ninja -S . -B "$(BUILD_DIR)" $(CMAKE_ARGS)
	ninja -C "$(BUILD_DIR)" -j "$(JOBS)"

test: test-lit test-python  ## Run everything

test-lit:                   ## MLIR FileCheck tests
	ninja -C "$(BUILD_DIR)" check-sar-lit

test-python:                ## Python frontend + backend tests
	PYTHONPATH=python:examples python3 -m pytest test/python -q
	bash test/test_fetch_llvm_release.sh

examples:                   ## Focus every synthetic example end-to-end
	cmake -E make_directory "$(ARTIFACT_DIR)/examples"
	PYTHONPATH=python python3 examples/wka/run_point_target_cpu.py --n 512 \
	  --output "$(ARTIFACT_DIR)/examples/wka.png"
	PYTHONPATH=python python3 examples/rda/run_point_target_cpu.py --n 512 \
	  --output "$(ARTIFACT_DIR)/examples/rda.png"
	PYTHONPATH=python python3 examples/csa/run_point_target_cpu.py --n 512 \
	  --output "$(ARTIFACT_DIR)/examples/csa.png"
	PYTHONPATH=python python3 examples/pfa/run_point_target_cpu.py --n 512 \
	  --output "$(ARTIFACT_DIR)/examples/pfa.png"

bench:                      ## Imaging-chain performance benchmarks
	PYTHONPATH=python python3 benchmarks/run_cpu_performance.py \
	  --sizes 1024 4096 --numpy

clean:
	@dir="$(abspath $(BUILD_DIR))"; \
	case "$$dir" in \
	  "$(CURDIR)"|"$(abspath $(CURDIR)/..)"|"/") \
	    echo "refusing to remove unsafe BUILD_DIR=$$dir" >&2; exit 1 ;; \
	  "$(CURDIR)"/*) rm -rf -- "$$dir" ;; \
	  *) echo "refusing to remove BUILD_DIR outside the repository: $$dir" >&2; \
	     exit 1 ;; \
	esac
