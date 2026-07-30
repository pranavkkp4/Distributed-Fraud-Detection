#include "fraud_engine/execution_engine.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>

namespace fraud_engine {
namespace {
constexpr int kFeatureCount = static_cast<int>(kFeatureWidth);
constexpr int kEmbeddingCount = static_cast<int>(kEmbeddingWidth);
constexpr int kContextCount = static_cast<int>(kMaxContextLength * kEmbeddingWidth);

__global__ void convert_fp16(const float* source, __half* destination, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) destination[index] = __float2half(source[index]);
}

__global__ void projection_fp16(const __half* input, const __half* weights,
                                const float* bias, float* output) {
  const int row = threadIdx.x;
  if (row >= kEmbeddingCount) return;
  float sum = bias[row];
  for (int column = 0; column < kFeatureCount; ++column)
    sum = fmaf(__half2float(weights[row * kFeatureCount + column]),
               __half2float(input[column]), sum);
  output[row] = sum;
}

__global__ void attention_fp16(const __half* query, const __half* context,
                               int context_length, float* output) {
  if (threadIdx.x != 0 || context_length < 0 || context_length > kMaxContextLength) return;
  float logits[kMaxContextLength];
  float maximum = -3.402823466e+38F;
  const int first_token = static_cast<int>(kMaxContextLength) - context_length;
  for (int token = 0; token < context_length; ++token) {
    float dot = 0.0F;
    for (int channel = 0; channel < kEmbeddingCount; ++channel)
      dot = fmaf(__half2float(query[channel]),
                 __half2float(context[(first_token + token) * kEmbeddingCount + channel]), dot);
    logits[token] = dot * 0.125F;
    maximum = fmaxf(maximum, logits[token]);
  }
  float denominator = 0.0F;
  for (int token = 0; token < context_length; ++token) {
    logits[token] = expf(logits[token] - maximum);
    denominator += logits[token];
  }
  for (int channel = 0; channel < kEmbeddingCount; ++channel) {
    float value = 0.0F;
    for (int token = 0; token < context_length; ++token)
      value = fmaf(logits[token] / denominator,
                   __half2float(context[(first_token + token) * kEmbeddingCount + channel]), value);
    output[channel] = value;
  }
}

__device__ float clip_e4m3(float value, float quant_scale) {
  return fminf(kFp8E4M3Max, fmaxf(-kFp8E4M3Max, value * quant_scale));
}

__global__ void convert_fp8(const float* source, __nv_fp8_e4m3* destination, int count,
                            float quant_scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    destination[index] = __nv_fp8_e4m3(clip_e4m3(source[index], quant_scale));
}

__global__ void projection_fp8(const __nv_fp8_e4m3* input,
                               const __nv_fp8_e4m3* weights, const float* bias,
                               float feature_dequant_scale, float weight_dequant_scale,
                               float* output) {
  const int row = threadIdx.x;
  if (row >= kEmbeddingCount) return;
  float sum = bias[row];
  for (int column = 0; column < kFeatureCount; ++column)
    sum = fmaf(static_cast<float>(weights[row * kFeatureCount + column]) *
                   weight_dequant_scale,
               static_cast<float>(input[column]) * feature_dequant_scale, sum);
  output[row] = sum;
}

__global__ void attention_fp8(const __nv_fp8_e4m3* query,
                              const __nv_fp8_e4m3* context, int context_length,
                              float query_dequant_scale, float context_dequant_scale,
                              float* output) {
  if (threadIdx.x != 0 || context_length < 0 || context_length > kMaxContextLength) return;
  float logits[kMaxContextLength];
  float maximum = -3.402823466e+38F;
  const int first_token = static_cast<int>(kMaxContextLength) - context_length;
  for (int token = 0; token < context_length; ++token) {
    float dot = 0.0F;
    for (int channel = 0; channel < kEmbeddingCount; ++channel)
      dot = fmaf(static_cast<float>(query[channel]) * query_dequant_scale,
                 static_cast<float>(context[(first_token + token) * kEmbeddingCount + channel]) *
                     context_dequant_scale,
                 dot);
    logits[token] = dot * 0.125F;
    maximum = fmaxf(maximum, logits[token]);
  }
  float denominator = 0.0F;
  for (int token = 0; token < context_length; ++token) {
    logits[token] = expf(logits[token] - maximum);
    denominator += logits[token];
  }
  for (int channel = 0; channel < kEmbeddingCount; ++channel) {
    float value = 0.0F;
    for (int token = 0; token < context_length; ++token)
      value = fmaf(logits[token] / denominator,
                   static_cast<float>(context[(first_token + token) * kEmbeddingCount + channel]) *
                       context_dequant_scale,
                   value);
    output[channel] = value;
  }
}

Status cuda_error_status(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return Status::Ok();
    case cudaErrorMemoryAllocation:
      return {StatusCode::kOutOfMemory, "CUDA microkernel allocation failed"};
    case cudaErrorNoDevice:
    case cudaErrorInvalidDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorDevicesUnavailable:
      return {StatusCode::kUnavailable, "CUDA runtime or current device is unavailable"};
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorUnsupportedPtxVersion:
      return {StatusCode::kUnsupported, "CUDA binary does not support the current device"};
    case cudaErrorLaunchFailure:
    case cudaErrorIllegalAddress:
    case cudaErrorAssert:
    case cudaErrorLaunchTimeout:
      return {StatusCode::kInternal, "CUDA microkernel execution fault"};
    default:
      return {StatusCode::kInternal, "CUDA microkernel runtime failure"};
  }
}

Status validate_runtime_backend(Backend backend) {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return cuda_error_status(error);
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return cuda_error_status(error);
  const int compute_capability = properties.major * 10 + properties.minor;
#ifndef FRAUD_ENGINE_MIN_CUDA_CC
#define FRAUD_ENGINE_MIN_CUDA_CC 75
#endif
#ifndef FRAUD_ENGINE_FP8_MIN_CUDA_CC
#define FRAUD_ENGINE_FP8_MIN_CUDA_CC 89
#endif
  if (compute_capability < FRAUD_ENGINE_MIN_CUDA_CC)
    return {StatusCode::kUnsupported, "current device is below the compiled CUDA architecture floor"};
  if (backend == Backend::kCudaFp8 && compute_capability < FRAUD_ENGINE_FP8_MIN_CUDA_CC)
    return {StatusCode::kUnsupported, "current device does not support E4M3 microkernels"};
  if (backend != Backend::kCudaFp8 && backend != Backend::kCudaFp16)
    return {StatusCode::kUnsupported, "requested backend is not a CUDA microkernel"};
  return Status::Ok();
}
}  // namespace

Status cuda_run_microkernel(const MicrokernelWeights& model,
                            const MicrokernelRequest& request, Backend backend,
                            MicrokernelResult* result) {
  if (result == nullptr)
    return {StatusCode::kInvalidArgument, "CUDA microkernel result is null"};
  const Status runtime_status = validate_runtime_backend(backend);
  if (!runtime_status.ok()) return runtime_status;

  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) return cuda_error_status(error);

  std::array<void*, 10> allocations{};
  std::size_t allocation_count = 0;
  bool async_allocations = false;
  int memory_pools_supported = 0;
  int current_device = 0;
  if (cudaGetDevice(&current_device) == cudaSuccess &&
      cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported,
                             current_device) == cudaSuccess)
    async_allocations = memory_pools_supported != 0;

  auto allocate = [&](void** pointer, std::size_t bytes) -> cudaError_t {
    const cudaError_t allocation_error = async_allocations
        ? cudaMallocAsync(pointer, bytes, stream)
        : cudaMalloc(pointer, bytes);
    if (allocation_error == cudaSuccess) allocations[allocation_count++] = *pointer;
    return allocation_error;
  };

  float* features_fp32 = nullptr;
  float* weights_fp32 = nullptr;
  float* context_fp32 = nullptr;
  float* bias = nullptr;
  float* projection = nullptr;
  float* attended = nullptr;
  void* features_lowp = nullptr;
  void* weights_lowp = nullptr;
  void* context_lowp = nullptr;
  void* query_lowp = nullptr;
  const std::size_t scalar_bytes =
      backend == Backend::kCudaFp8 ? sizeof(__nv_fp8_e4m3) : sizeof(__half);

  auto run = [&]() -> cudaError_t {
    cudaError_t current =
        allocate(reinterpret_cast<void**>(&features_fp32), sizeof(float) * kFeatureCount);
    if (current != cudaSuccess) return current;
    current = allocate(reinterpret_cast<void**>(&weights_fp32),
                       sizeof(float) * kFeatureCount * kEmbeddingCount);
    if (current != cudaSuccess) return current;
    current = allocate(reinterpret_cast<void**>(&context_fp32), sizeof(float) * kContextCount);
    if (current != cudaSuccess) return current;
    current = allocate(reinterpret_cast<void**>(&bias), sizeof(float) * kEmbeddingCount);
    if (current != cudaSuccess) return current;
    current = allocate(reinterpret_cast<void**>(&projection), sizeof(float) * kEmbeddingCount);
    if (current != cudaSuccess) return current;
    current = allocate(reinterpret_cast<void**>(&attended), sizeof(float) * kEmbeddingCount);
    if (current != cudaSuccess) return current;
    current = allocate(&features_lowp, scalar_bytes * kFeatureCount);
    if (current != cudaSuccess) return current;
    current = allocate(&weights_lowp, scalar_bytes * kFeatureCount * kEmbeddingCount);
    if (current != cudaSuccess) return current;
    current = allocate(&context_lowp, scalar_bytes * kContextCount);
    if (current != cudaSuccess) return current;
    current = allocate(&query_lowp, scalar_bytes * kEmbeddingCount);
    if (current != cudaSuccess) return current;

    current = cudaMemcpyAsync(features_fp32, request.transaction_features.data(),
                              sizeof(float) * kFeatureCount, cudaMemcpyHostToDevice, stream);
    if (current != cudaSuccess) return current;
    current = cudaMemcpyAsync(weights_fp32, model.projection_weights.data(),
                              sizeof(float) * kFeatureCount * kEmbeddingCount,
                              cudaMemcpyHostToDevice, stream);
    if (current != cudaSuccess) return current;
    current = cudaMemcpyAsync(context_fp32, request.context.data(), sizeof(float) * kContextCount,
                              cudaMemcpyHostToDevice, stream);
    if (current != cudaSuccess) return current;
    current = cudaMemcpyAsync(bias, model.projection_bias.data(), sizeof(float) * kEmbeddingCount,
                              cudaMemcpyHostToDevice, stream);
    if (current != cudaSuccess) return current;

    constexpr int threads = 128;
    const int weight_blocks = (kFeatureCount * kEmbeddingCount + threads - 1) / threads;
    const int context_blocks = (kContextCount + threads - 1) / threads;
    if (backend == Backend::kCudaFp8) {
      convert_fp8<<<1, threads, 0, stream>>>(
          features_fp32, static_cast<__nv_fp8_e4m3*>(features_lowp), kFeatureCount,
          request.feature_fp8.quant_scale);
      convert_fp8<<<weight_blocks, threads, 0, stream>>>(
          weights_fp32, static_cast<__nv_fp8_e4m3*>(weights_lowp),
          kFeatureCount * kEmbeddingCount, model.weight_fp8.quant_scale);
      convert_fp8<<<context_blocks, threads, 0, stream>>>(
          context_fp32, static_cast<__nv_fp8_e4m3*>(context_lowp), kContextCount,
          request.context_fp8.quant_scale);
      projection_fp8<<<1, kEmbeddingCount, 0, stream>>>(
          static_cast<__nv_fp8_e4m3*>(features_lowp),
          static_cast<__nv_fp8_e4m3*>(weights_lowp), bias,
          request.feature_fp8.dequant_scale, model.weight_fp8.dequant_scale, projection);
      convert_fp8<<<1, threads, 0, stream>>>(
          projection, static_cast<__nv_fp8_e4m3*>(query_lowp), kEmbeddingCount,
          model.projection_fp8.quant_scale);
      attention_fp8<<<1, 1, 0, stream>>>(
          static_cast<__nv_fp8_e4m3*>(query_lowp),
          static_cast<__nv_fp8_e4m3*>(context_lowp),
          static_cast<int>(request.context_length), model.projection_fp8.dequant_scale,
          request.context_fp8.dequant_scale, attended);
    } else {
      convert_fp16<<<1, threads, 0, stream>>>(
          features_fp32, static_cast<__half*>(features_lowp), kFeatureCount);
      convert_fp16<<<weight_blocks, threads, 0, stream>>>(
          weights_fp32, static_cast<__half*>(weights_lowp),
          kFeatureCount * kEmbeddingCount);
      convert_fp16<<<context_blocks, threads, 0, stream>>>(
          context_fp32, static_cast<__half*>(context_lowp), kContextCount);
      projection_fp16<<<1, kEmbeddingCount, 0, stream>>>(
          static_cast<__half*>(features_lowp), static_cast<__half*>(weights_lowp),
          bias, projection);
      convert_fp16<<<1, threads, 0, stream>>>(
          projection, static_cast<__half*>(query_lowp), kEmbeddingCount);
      attention_fp16<<<1, 1, 0, stream>>>(
          static_cast<__half*>(query_lowp), static_cast<__half*>(context_lowp),
          static_cast<int>(request.context_length), attended);
    }
    current = cudaGetLastError();
    if (current != cudaSuccess) return current;
    current = cudaMemcpyAsync(result->projection.data(), projection,
                              sizeof(float) * kEmbeddingCount, cudaMemcpyDeviceToHost, stream);
    if (current != cudaSuccess) return current;
    current = cudaMemcpyAsync(result->attended_context.data(), attended,
                              sizeof(float) * kEmbeddingCount, cudaMemcpyDeviceToHost, stream);
    if (current != cudaSuccess) return current;
    return cudaStreamSynchronize(stream);
  };

  error = run();
  cudaError_t cleanup_error = cudaSuccess;
  for (std::size_t index = 0; index < allocation_count; ++index) {
    const cudaError_t current = async_allocations
        ? cudaFreeAsync(allocations[index], stream)
        : cudaFree(allocations[index]);
    if (cleanup_error == cudaSuccess && current != cudaSuccess) cleanup_error = current;
  }
  if (async_allocations) {
    const cudaError_t synchronization_error = cudaStreamSynchronize(stream);
    if (cleanup_error == cudaSuccess) cleanup_error = synchronization_error;
  }
  const cudaError_t destroy_error = cudaStreamDestroy(stream);
  if (error == cudaSuccess) error = cleanup_error;
  if (error == cudaSuccess) error = destroy_error;
  return cuda_error_status(error);
}
}  // namespace fraud_engine
