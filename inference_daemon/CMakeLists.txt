cmake_minimum_required(VERSION 3.24)
project(fraud_inference_daemon LANGUAGES CXX)

include(CTest)
include(FetchContent)

option(FRAUD_DAEMON_ENABLE_CUDA "Build CUDA graph and CUTLASS integrations" OFF)
option(FRAUD_DAEMON_ENABLE_CUTLASS "Build the CUTLASS INT8 GEMM wrapper" OFF)

# The daemon verifies the Transaction FlatBuffer in-place.  FlatBuffers is
# header-only for this use; pin the source archive and its content hash rather
# than accepting an ambient system package or an unpinned generated header.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_GRPCTEST OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(flatbuffers
  URL https://github.com/google/flatbuffers/archive/refs/tags/v25.12.19.tar.gz
  URL_HASH SHA256=f81c3162b1046fe8b84b9a0dbdd383e24fdbcf88583b9cb6028f90d04d90696a)
FetchContent_MakeAvailable(flatbuffers)

set(FRAUD_DAEMON_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../api_gateway/src/generated")

add_library(fraud_daemon_core
  src/memory_pool.cpp
  src/batcher.cpp
  src/shared_memory.cpp
  src/transaction.cpp
  src/cuda_stubs.cpp)
target_include_directories(fraud_daemon_core PUBLIC include PRIVATE "${FRAUD_DAEMON_GENERATED_DIR}")
target_compile_features(fraud_daemon_core PUBLIC cxx_std_20)
target_link_libraries(fraud_daemon_core PRIVATE FlatBuffers::FlatBuffers)
target_compile_definitions(fraud_daemon_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)
if(MSVC)
  target_compile_options(fraud_daemon_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(fraud_daemon_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(fraud_inference_daemon src/main.cpp)
target_link_libraries(fraud_inference_daemon PRIVATE fraud_daemon_core)

if(FRAUD_DAEMON_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(fraud_daemon_core PRIVATE FRAUD_DAEMON_CUDA_LINKED=1)
  add_library(fraud_daemon_cuda src/cuda_graph.cu)
  target_include_directories(fraud_daemon_cuda PUBLIC include)
  target_compile_features(fraud_daemon_cuda PUBLIC cxx_std_20)
  target_link_libraries(fraud_daemon_cuda PUBLIC CUDA::cudart)
  set_target_properties(fraud_daemon_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
  if(MSVC)
    target_compile_options(fraud_daemon_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  endif()
  if(FRAUD_DAEMON_ENABLE_CUTLASS)
    if(NOT DEFINED CUTLASS_PATH)
      message(FATAL_ERROR "Set CUTLASS_PATH when FRAUD_DAEMON_ENABLE_CUTLASS=ON")
    endif()
    target_sources(fraud_daemon_cuda PRIVATE src/cutlass_int8_gemm.cu)
    target_include_directories(fraud_daemon_cuda PRIVATE "${CUTLASS_PATH}/include" "${CUTLASS_PATH}/tools/util/include")
    target_compile_definitions(fraud_daemon_cuda PUBLIC FRAUD_DAEMON_HAVE_CUTLASS=1)
    target_compile_definitions(fraud_daemon_core PRIVATE FRAUD_DAEMON_HAVE_CUTLASS=1)
    if(BUILD_TESTING)
      add_executable(fraud_daemon_gpu_benchmark tests/gpu_benchmark.cu)
      target_include_directories(fraud_daemon_gpu_benchmark PRIVATE include)
      target_link_libraries(fraud_daemon_gpu_benchmark PRIVATE fraud_daemon_cuda)
      set_target_properties(fraud_daemon_gpu_benchmark PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
      add_test(NAME fraud_daemon_gpu_benchmark COMMAND fraud_daemon_gpu_benchmark)
    endif()
  endif()
  target_link_libraries(fraud_inference_daemon PRIVATE fraud_daemon_cuda)
endif()

if(BUILD_TESTING)
  add_executable(fraud_daemon_tests tests/daemon_tests.cpp)
  target_include_directories(fraud_daemon_tests PRIVATE "${FRAUD_DAEMON_GENERATED_DIR}")
  target_link_libraries(fraud_daemon_tests PRIVATE fraud_daemon_core FlatBuffers::FlatBuffers)
  target_compile_features(fraud_daemon_tests PRIVATE cxx_std_20)
  add_test(NAME fraud_daemon_tests COMMAND fraud_daemon_tests)
endif()
