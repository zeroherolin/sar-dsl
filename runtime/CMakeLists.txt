# libsar_runtime: standalone C-ABI kernels (FFT, Stolt interpolation) called
# from CPU-compiled SAR kernels. Deliberately has no LLVM/MLIR dependency.

add_library(sar_runtime SHARED SARRuntime.cpp)
set_target_properties(sar_runtime PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
# Only the project's own headers; RuntimeEnums.h is plain C++ by design.
target_include_directories(sar_runtime PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(sar_runtime PRIVATE
  $<$<NOT:$<CONFIG:Debug>>:-O3> -fno-exceptions)
find_package(Threads REQUIRED)
target_link_libraries(sar_runtime PRIVATE Threads::Threads)
