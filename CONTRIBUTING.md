# Contributing to SAR-DSL

## Development setup

```bash
git submodule update --init externals/llvm-project
make llvm        # one-time LLVM/MLIR/Clang build
make dev         # project build + backend symlinks
export PYTHONPATH=$PWD/python:$PYTHONPATH
make test        # lit + pytest, everything must pass
```

`make scalehls` additionally builds the ScaleHLS-HIDA toolchain (needed only
for the `scalehls` backend and its tests; they self-skip otherwise).

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
  declared in tablegen (`include/sar/Conversion/Passes.td`) and
  registered in `sar-opt`.
- **Python** is PEP 8 with 79-column lines (enforced by yapf + ruff);
  keep the frontend dependency-free beyond numpy.
- **Tests are mandatory.** Dialect/lowering changes need lit coverage under
  `test/Dialect` / `test/Conversion`; user-visible behavior needs pytest
  coverage under `test/python`. Numerical kernels are validated against
  numpy references with explicit tolerances.
- **Folds must be bit-exact.** Rewrites that can change floating-point
  results (even by one ULP) do not belong in canonicalization.

## Adding functionality

- *New SAR operation*: ODS definition in `SAROps.td` (+ verifier), lowering
  pattern in `lib/Conversion/`, frontend wrapper in `python/sar/language`,
  lit + pytest coverage, and a row in `docs/dialect.md`.
- *New backend*: see `docs/backends.md`; nothing in the core should change.
- *Runtime kernels*: `runtime/SARRuntime.cpp` is dependency-free C++17 with
  a C ABI; keep it that way.

## ScaleHLS-HIDA development

The submodule tracks the `dev` branch of
[our fork](https://github.com/zeroherolin/ScaleHLS-HIDA), ported to the
project's LLVM tree. Fix HIDA bugs there as regular commits, push the
branch, then bump the submodule pointer here -- no patch files.
