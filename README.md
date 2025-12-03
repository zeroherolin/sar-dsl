# MLIR-based SAR Domain-Specific Accelerator Lang

The project is currently under development...

## Getting Started

### Instructions

- Clone repository

```bash
git clone https://github.com/zeroherolin/sar-dsl.git
cd sar-dsl && git submodule update --init --recursive
```

- Requirements

```
sudo apt install -y cmake clang lld

conda create -n sar-dsl python=3.12
conda activate sar-dsl
pip install -y pybind11 nanobind
```

- Build MLIR

```bash
cd externals/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python)

ninja
```

- Build ScaleHLS

Duplicated MLIR is built here cuz' ScaleHLS depends on the old version.

```bash
cd ../../ScaleHLS-HIDA/polygeist/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja

cd ../../..  # pwd: externals/ScaleHLS-HIDA

sed -i '18 a\
set(LLVM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/polygeist/llvm-project/build")\
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm")\
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")\
' ./CMakeLists.txt

mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja scalehls-opt scalehls-translate
```

- Build SAR-DSL

```bash
cd ../../..  # pwd: sar-dsl/
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

ninja
```

- Run tests

```bash
cd ..  # pwd: sar-dsl/
bash run_flow.sh
```
