cmake_minimum_required(VERSION 3.20)
project(fraud_microkernel_pipeline LANGUAGES CXX)

option(FRAUD_ENGINE_ENABLE_CUDA "Build optional CUDA capability probes and kernels" ON)
option(FRAUD_ENGINE_BUILD_TESTS "Build microkernel-pipeline tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FRAUD_ENGINE_SOURCES
  src/execution_engine.cpp
  src/graph/graph_replay.cpp
  src/kernels/reference_kernels.cpp
  src/memory/stream_ordered_arena.cpp)

set(FRAUD_ENGINE_HAS_CUDA OFF)
if(FRAUD_ENGINE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND FRAUD_ENGINE_SOURCES src/cuda_runtime.cu src/kernels/cuda_kernels.cu src/memory/cuda_arena.cu)
      set(FRAUD_ENGINE_HAS_CUDA ON)
    else()
      message(STATUS "CUDA backend disabled: CUDA 11.8 or newer is required")
    endif()
  endif()
endif()

add_library(fraud_microkernel_pipeline ${FRAUD_ENGINE_SOURCES})
add_library(fraud_execution_engine::microkernel_pipeline ALIAS fraud_microkernel_pipeline)
target_include_directories(fraud_microkernel_pipeline PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(fraud_microkernel_pipeline PUBLIC cxx_std_17)
target_compile_definitions(fraud_microkernel_pipeline PRIVATE
  FRAUD_ENGINE_MIN_CUDA_CC=75
  FRAUD_ENGINE_FP8_MIN_CUDA_CC=89
  $<$<BOOL:${FRAUD_ENGINE_HAS_CUDA}>:FRAUD_ENGINE_WITH_CUDA>)
if(MSVC)
  # Do not forward MSVC-only switches to nvcc when CUDA is enabled.
  target_compile_options(fraud_microkernel_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
  target_compile_options(fraud_microkernel_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()
if(FRAUD_ENGINE_HAS_CUDA)
  set_target_properties(fraud_microkernel_pipeline PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES "75;80;86;89")
endif()

if(FRAUD_ENGINE_BUILD_TESTS)
  enable_testing()
  add_executable(fraud_microkernel_tests ../tests/unit/execution_engine_tests.cpp)
  target_link_libraries(fraud_microkernel_tests PRIVATE fraud_microkernel_pipeline)
  add_test(NAME fraud_microkernel_tests COMMAND fraud_microkernel_tests)
endif()
