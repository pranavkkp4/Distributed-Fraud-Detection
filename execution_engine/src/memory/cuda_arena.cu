#include "fraud_engine/execution_engine.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <new>

namespace fraud_engine {
namespace {
struct CudaArenaContext {
  cudaStream_t stream{nullptr};
  int device{-1};
  cudaError_t pending_error{cudaSuccess};
};

Status arena_status(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return Status::Ok();
    case cudaErrorMemoryAllocation:
      return {StatusCode::kOutOfMemory, "CUDA arena allocation failed"};
    case cudaErrorNoDevice:
    case cudaErrorInvalidDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorDevicesUnavailable:
      return {StatusCode::kUnavailable, "CUDA arena device is unavailable"};
    default:
      return {StatusCode::kInternal, "CUDA arena stream operation failed"};
  }
}

void remember(CudaArenaContext* context, cudaError_t error) {
  if (context->pending_error == cudaSuccess && error != cudaSuccess)
    context->pending_error = error;
}

cudaError_t activate(CudaArenaContext* context, int* previous_device) {
  cudaError_t error = cudaGetDevice(previous_device);
  if (error == cudaSuccess && *previous_device != context->device)
    error = cudaSetDevice(context->device);
  remember(context, error);
  return error;
}

void restore(CudaArenaContext* context, int previous_device) {
  if (previous_device != context->device) remember(context, cudaSetDevice(previous_device));
}
}  // namespace

void* cuda_arena_create() {
#if CUDART_VERSION < 11020
  return nullptr;
#else
  int device = 0;
  int memory_pools_supported = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device) !=
          cudaSuccess ||
      memory_pools_supported == 0) return nullptr;
  auto* context = new (std::nothrow) CudaArenaContext();
  if (context == nullptr) return nullptr;
  context->device = device;
  if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) {
    delete context;
    return nullptr;
  }
  return context;
#endif
}

Status cuda_arena_allocate(void* opaque_context, std::size_t bytes, void** result) {
  if (opaque_context == nullptr || result == nullptr)
    return {StatusCode::kInvalidArgument, "CUDA arena allocation arguments are invalid"};
  auto* context = static_cast<CudaArenaContext*>(opaque_context);
  int previous_device = context->device;
  cudaError_t error = activate(context, &previous_device);
  *result = nullptr;
#if CUDART_VERSION >= 11020
  if (error == cudaSuccess) error = cudaMallocAsync(result, bytes, context->stream);
#else
  (void)bytes;
  error = cudaErrorNotSupported;
#endif
  remember(context, error);
  restore(context, previous_device);
  return arena_status(error);
}

Status cuda_arena_free(void* opaque_context, void* memory) noexcept {
  if (opaque_context == nullptr || memory == nullptr) return Status::Ok();
  auto* context = static_cast<CudaArenaContext*>(opaque_context);
  int previous_device = context->device;
  cudaError_t error = activate(context, &previous_device);
#if CUDART_VERSION >= 11020
  if (error == cudaSuccess) error = cudaFreeAsync(memory, context->stream);
#endif
  remember(context, error);
  restore(context, previous_device);
  return arena_status(error);
}

Status cuda_arena_synchronize(void* opaque_context) noexcept {
  if (opaque_context == nullptr) return Status::Ok();
  auto* context = static_cast<CudaArenaContext*>(opaque_context);
  int previous_device = context->device;
  cudaError_t error = activate(context, &previous_device);
#if CUDART_VERSION >= 11020
  if (error == cudaSuccess) error = cudaStreamSynchronize(context->stream);
#endif
  remember(context, error);
  const cudaError_t retained_error = context->pending_error;
  context->pending_error = cudaSuccess;
  restore(context, previous_device);
  if (retained_error != cudaSuccess) return arena_status(retained_error);
  return arena_status(context->pending_error);
}

void* cuda_arena_stream(void* opaque_context) noexcept {
  if (opaque_context == nullptr) return nullptr;
  return static_cast<CudaArenaContext*>(opaque_context)->stream;
}

int cuda_arena_device(void* opaque_context) noexcept {
  if (opaque_context == nullptr) return -1;
  return static_cast<CudaArenaContext*>(opaque_context)->device;
}

void cuda_arena_destroy(void* opaque_context) noexcept {
  if (opaque_context == nullptr) return;
  auto* context = static_cast<CudaArenaContext*>(opaque_context);
  int previous_device = context->device;
  if (activate(context, &previous_device) == cudaSuccess) {
    (void)cudaStreamSynchronize(context->stream);
    (void)cudaStreamDestroy(context->stream);
  }
  restore(context, previous_device);
  delete context;
}
}  // namespace fraud_engine
