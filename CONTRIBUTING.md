# Contributing to SAR-DSL

## Development setup

```bash
git submodule update --init externals/llvm-project
make llvm        # one-time LLVM/MLIR/Clang build
make build       # sar-opt, sar-translate, runtime, tests
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
make test        # lit + pytest, everything must pass
```

`make build` writes `build/python/sar/_build_config.py`, which points the Python package at the active build. For a non-default directory, build with `make BUILD_DIR=build-local` and set `SAR_DSL_BUILD_DIR=build-local` when running Python. Individual tools can be overridden with `SAR_DSL_TOOL_<NAME>`. See [docs/backends.md](docs/backends.md) for the discovery order.

The GitHub Actions compiler job downloads the pinned LLVM 22.1.8 Linux release and builds only SAR-DSL against it. The local `make llvm` path remains the source-build option for editor integration and LLVM changes.

To reproduce the CI toolchain locally, run `scripts/fetch-llvm-release.sh`, then pass its reported directory as `MLIR_DIR`, `LLVM_DIR`, `SAR_DSL_LLVM_BUILD_DIR`, and `SAR_DSL_LLVM_TOOL_DIR` to CMake.

## Editor setup

`make llvm` builds clangd, `make build` generates the compilation database and the `sar-lsp-server` (the MLIR language server with the `sar` dialect registered -- the upstream one would flag every `sar` op as unknown); the committed `.clangd` and `.vscode/` wire them up (plus the TableGen LSP server), so navigation and diagnostics work out of the box in VS Code. Other editors: use the in-tree `externals/llvm-project/build/bin/clangd` and `build/bin/sar-lsp-server`.

## Formatting

Formatting is enforced by [pre-commit](https://pre-commit.com): clang-format (LLVM style) for C++, yapf + ruff (pep8, 79 columns) for Python, plus whitespace/file hygiene hooks.

```bash
python -m pip install pre-commit
pre-commit install         # run on every commit
pre-commit run --all-files # run manually
```

CI runs the same hooks on pull requests and the main branch.

## Repository conventions

- **C++** follows the LLVM style (2-space indent, 80 columns, `lowerCamelCase` functions; enforced by clang-format). New passes are declared in TableGen: conversions in `include/sar/Conversion/Passes.td`, SAR transforms in `include/sar/Dialect/SAR/Transforms/Passes.td`, and HLS transforms in `include/sar/Dialect/HLS/Transforms/Passes.td`. Register them in `sar-opt`.
- **Python** is PEP 8 with 79-column lines (enforced by yapf + ruff); keep the frontend dependency-free beyond NumPy.
- **Markdown** prose is not hard-wrapped; keep each paragraph or list item on one source line and let the renderer wrap it. Preserve structural line breaks in headings, tables, lists, fenced code, and HTML blocks.
- **Tests are mandatory.** Dialect/lowering changes need lit coverage under `test/Dialect` / `test/Conversion`; user-visible behavior needs pytest coverage under `test/python`. Numerical kernels are validated against NumPy references with explicit tolerances. Changes to the Stockham or Bluestein lowering must also be checked numerically on CPU through `--sar-affine-to-llvm-pipeline` (see `test/python/test_hls_flow.py`), since that path is what the HLS backend emits from.
- **Every backend runs every kernel.** The DSL must not grow a construct that only one backend can lower; `test/python/test_backend_symmetry.py` is the gate.
- **Folds must be bit-exact.** Rewrites that can change floating-point results (even by one ULP) do not belong in canonicalization.

## Adding functionality

- _New SAR operation_: ODS definition in `SAROps.td` (+ verifier), lowering pattern in `lib/Conversion/`, frontend wrapper in `python/sar/language`, lit + pytest coverage, and a row in `docs/dialect.md`.
- _New backend_: see `docs/backends.md`; nothing in the core should change.
- _Runtime kernels_: `runtime/SARRuntime.cpp` is dependency-free C++17 with a C ABI; keep it that way.

## HLS dialect development

The `hls` dialect and its passes (`include/sar/Dialect/HLS`, `lib/Dialect/HLS`, `lib/Target/HLS`) are ordinary in-tree code: change them like any other pass, and `test/Dialect/HLS/*.mlir` covers them.
