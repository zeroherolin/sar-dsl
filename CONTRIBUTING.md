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

C++ navigation works through clangd with the compilation database the
build generates (`build/compile_commands.json`); the committed `.clangd`
points clangd at it, so any clangd-based editor works after `make build`.
For VS Code, `.vscode/extensions.json` recommends the clangd and MLIR
extensions and `.vscode/settings.json` wires up the TableGen/MLIR/PDLL
language servers from the in-tree LLVM build (`.td`/`.mlir` files get
navigation and diagnostics too). All committed editor settings use
repository-relative paths only.

## Repository conventions

- **C++** follows the LLVM style (2-space indent, 80 columns,
  `lowerCamelCase` functions). New passes are declared in tablegen
  (`include/sar/Conversion/Passes.td`) and registered in `sar-opt`.
- **Python** is PEP 8 with 79-column lines; keep the frontend dependency-free
  beyond numpy. `python -m pyflakes python/sar third_party test/python
  examples/*/*.py benchmarks/*.py` must be clean.
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

## Submodule patches

Local fixes to `externals/` must be captured as patch files under
`scripts/patches/` and applied idempotently from the build scripts, so a
fresh clone reproduces the toolchain exactly.
