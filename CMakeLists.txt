# ===----------------------------------------------------------------------===
# SAR-DSL: an MLIR-based compiler for SAR imaging pipelines.
# ===----------------------------------------------------------------------===

cmake_minimum_required(VERSION 3.20)
project(sar-dsl LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# ------------------------------------------------------------------------- #
# MLIR / LLVM
#
# By default the in-tree submodule build is used; override MLIR_DIR / LLVM_DIR
# to build against an external installation.
# ------------------------------------------------------------------------- #
set(SAR_DSL_LLVM_BUILD_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/build"
    CACHE PATH "LLVM build/install tree providing LLVM and MLIR")

if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "${SAR_DSL_LLVM_BUILD_DIR}/lib/cmake/mlir")
endif()
if(NOT DEFINED LLVM_DIR)
  set(LLVM_DIR "${SAR_DSL_LLVM_BUILD_DIR}/lib/cmake/llvm")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set(_sar_default_llvm_tools "${LLVM_TOOLS_BINARY_DIR}")
if(NOT _sar_default_llvm_tools)
  set(_sar_default_llvm_tools "${SAR_DSL_LLVM_BUILD_DIR}/bin")
endif()
set(SAR_DSL_LLVM_TOOL_DIR "${_sar_default_llvm_tools}" CACHE PATH
    "Directory containing LLVM tools used by the Python driver")
unset(_sar_default_llvm_tools)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
# LLVM's release package exports the C++ ABI macro both through
# `LLVM_DEFINITIONS` and `llvm_update_compile_flags`. Keep the shared set of
# definitions, but let the target helper provide this macro once.
set(_sar_llvm_definitions ${LLVM_DEFINITIONS})
list(FILTER _sar_llvm_definitions EXCLUDE REGEX "_GLIBCXX_USE_CXX11_ABI")
add_definitions(${_sar_llvm_definitions})
unset(_sar_llvm_definitions)

# ------------------------------------------------------------------------- #
# Subdirectories
# ------------------------------------------------------------------------- #
add_subdirectory(include/sar)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(tools)

install(TARGETS sar-opt sar-translate sar-lsp-server sar_runtime
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/sar/Runtime/RuntimeEnums.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sar/Runtime)
install(FILES LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/sar-dsl)

option(SAR_DSL_ENABLE_TESTS "Build SAR-DSL tests" ON)
if(SAR_DSL_ENABLE_TESTS)
  add_subdirectory(test)
endif()

# ------------------------------------------------------------------------- #
# Python build configuration
#
# Generates python/sar/_build_config.py so the Python driver can locate the
# freshly built tools without any environment setup.
# ------------------------------------------------------------------------- #
set(SAR_DSL_TOOL_DIR "${PROJECT_BINARY_DIR}/bin")
set(SAR_DSL_RUNTIME_LIB "${PROJECT_BINARY_DIR}/lib/libsar_runtime.so")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/python/sar")
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/_build_config.py.in
  ${PROJECT_BINARY_DIR}/python/sar/_build_config.py
  @ONLY)

# Editor LSP servers referenced by .vscode/settings.json.
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
foreach(tool tblgen-lsp-server mlir-pdll-lsp-server)
  if(EXISTS "${SAR_DSL_LLVM_TOOL_DIR}/${tool}")
    file(CREATE_LINK "${SAR_DSL_LLVM_TOOL_DIR}/${tool}"
         "${PROJECT_BINARY_DIR}/bin/${tool}" SYMBOLIC)
  endif()
endforeach()
