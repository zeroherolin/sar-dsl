# Contributing to SAR-DSL

## Development setup

```bash
git submodule update --init externals/llvm-project
make llvm        # one-time LLVM/MLIR/Clang build
make build       # sar-opt, sar-translate, runtime, tests
export PYTHONPATH=$PWD/python:$PYTHONPATH
make test        # lit + pytest, everything must pass
```

`make build` writes `python/sar/_build_config.py`, which points the Python
package at `build/bin`. If the tools move, or you want to run against a
different build tree, `SAR_DSL_TOOL_DIR` overrides that directory; see
[docs/backends.md](docs/backends.md) for the per-tool overrides.

## Editor setup

`make llvm` builds clangd, `make build` generates the compilation
database and the `sar-lsp-server` (the MLIR language server with the
`sar` dialect registered -- the upstream one would flag every `sar` op
as unknown); the committed `.clangd` and `.vscode/` wire them up (plus
the TableGen LSP server), so navigation and diagnostics work out of
the box in VS Code. Other editors: use the in-tree
`externals/llvm-project/build/bin/clangd` and `build/bin/sar-lsp-server`.

## Formatting

Formatting is enforced by [pre-commit](https://pre-commit.com):
clang-format (LLVM style) for C++, yapf + ruff (pep8, 79 columns) for
Python, plus whitespace/file hygiene hooks.

```bash
pip install pre-commit
pre-commit install         # run on every commit
pre-commit run --all-files # run manually
```

CI runs the same hooks on every push.

## Repository conventions

- **C++** follows the LLVM style (2-space indent, 80 columns,
  `lowerCamelCase` functions; enforced by clang-format). New passes are
  declared in tablegen -- conversions in `include/sar/Conversion/Passes.td`,
  dialect transforms in `include/sar/Dialect/SAR/Transforms/Passes.td` --
  and registered in `sar-opt`.
- **Python** is PEP 8 with 79-column lines (enforced by yapf + ruff);
  keep the frontend dependency-free beyond numpy.
- **Tests are mandatory.** Dialect/lowering changes need lit coverage under
  `test/Dialect` / `test/Conversion`; user-visible behavior needs pytest
  coverage under `test/python`. Numerical kernels are validated against
  numpy references with explicit tolerances. Changes to the Stockham or
  Bluestein lowering must also be checked numerically on CPU through
  `--sar-affine-to-llvm-pipeline` (see `test/python/test_hls_flow.py`),
  since that path is what the HLS backend emits from.
- **Every backend runs every kernel.** The DSL must not grow a construct
  that only one backend can lower; `test/python/test_backend_symmetry.py`
  is the gate.
- **Folds must be bit-exact.** Rewrites that can change floating-point
  results (even by one ULP) do not belong in canonicalization.

## Adding functionality

- *New SAR operation*: ODS definition in `SAROps.td` (+ verifier), lowering
  pattern in `lib/Conversion/`, frontend wrapper in `python/sar/language`,
  lit + pytest coverage, and a row in `docs/dialect.md`.
- *New backend*: see `docs/backends.md`; nothing in the core should change.
- *Runtime kernels*: `runtime/SARRuntime.cpp` is dependency-free C++17 with
  a C ABI; keep it that way.

## HLS dialect development

The `hls` dialect and its passes (`include/sar/Dialect/HLS`,
`lib/Dialect/HLS`, `lib/Target/HLS`) are ordinary in-tree code: change
them like any other pass, and `test/Dialect/HLS/*.mlir` covers them.
They derive from ScaleHLS-HIDA -- see [NOTICE](NOTICE) -- so files
carrying the original copyright header keep it.
