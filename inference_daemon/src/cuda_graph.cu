#include "fraud_daemon/cuda_graph.hpp"

#include <cuda_runtime.h>
#include <array>
#include <mutex>

namespace fraud::daemon {
namespace {
constexpr std::size_t kMaxBatch = 256;
struct GraphSlot { cudaGraphExec_t exec{}; cudaGraph_t graph{}; cudaStream_t stream{}; float* host{}; float* device{}; };
std::array<GraphSlot, kMaxBatch + 1> graphs{};
std::mutex graph_mutex;
__global__ void forward_stub(float* values, std::size_t n) { const auto i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) values[i] = values[i] * 1.0F; }
CudaGraphStatus status(cudaError_t result, const char* action) noexcept { return result == cudaSuccess ? CudaGraphStatus{CudaCapability::ready, "ready"} : CudaGraphStatus{CudaCapability::error, action}; }
}
CudaGraphStatus capture_inference_graph(std::size_t batch_size) noexcept {
  if (batch_size == 0 || batch_size > kMaxBatch) return {CudaCapability::error, "unsupported batch size"};
  std::lock_guard<std::mutex> lock(graph_mutex); auto& slot = graphs[batch_size];
  if (slot.exec != nullptr) return {CudaCapability::ready, "already captured"};
  if (auto result = cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking); result != cudaSuccess) return status(result, "cudaStreamCreate failed");
  const auto bytes = batch_size * sizeof(float);
  if (auto result = cudaHostAlloc(&slot.host, bytes, cudaHostAllocDefault); result != cudaSuccess) { cudaStreamDestroy(slot.stream); slot.stream = {}; return status(result, "cudaHostAlloc failed"); }
  if (auto result = cudaMalloc(&slot.device, bytes); result != cudaSuccess) { cudaFreeHost(slot.host); slot.host = nullptr; cudaStreamDestroy(slot.stream); slot.stream = {}; return status(result, "cudaMalloc failed"); }
  cudaError_t result = cudaStreamBeginCapture(slot.stream, cudaStreamCaptureModeGlobal);
  if (result == cudaSuccess) result = cudaMemcpyAsync(slot.device, slot.host, bytes, cudaMemcpyHostToDevice, slot.stream);
  if (result == cudaSuccess) { forward_stub<<<static_cast<unsigned>((batch_size + 255) / 256), 256, 0, slot.stream>>>(slot.device, batch_size); result = cudaGetLastError(); }
  if (result == cudaSuccess) result = cudaMemcpyAsync(slot.host, slot.device, bytes, cudaMemcpyDeviceToHost, slot.stream);
  if (result == cudaSuccess) result = cudaStreamEndCapture(slot.stream, &slot.graph); else { cudaGraph_t discarded{}; (void)cudaStreamEndCapture(slot.stream, &discarded); if (discarded) cudaGraphDestroy(discarded); }
  if (result == cudaSuccess) result = cudaGraphInstantiate(&slot.exec, slot.graph, nullptr, nullptr, 0);
  if (result != cudaSuccess) { if (slot.exec) cudaGraphExecDestroy(slot.exec); if (slot.graph) cudaGraphDestroy(slot.graph); if (slot.device) cudaFree(slot.device); if (slot.host) cudaFreeHost(slot.host); if (slot.stream) cudaStreamDestroy(slot.stream); slot = {}; }
  return status(result, "CUDA Graph capture/instantiate failed");
}
CudaGraphStatus replay_inference_graph(std::size_t batch_size) noexcept {
  const auto launched = launch_inference_graph(batch_size);
  if (launched.capability != CudaCapability::ready) return launched;
  return synchronize_inference_graph(batch_size);
}
CudaGraphStatus launch_inference_graph(std::size_t batch_size) noexcept {
  if (batch_size == 0 || batch_size > kMaxBatch || graphs[batch_size].exec == nullptr) return {CudaCapability::unavailable, "batch size was not captured at startup"};
  const auto& slot = graphs[batch_size];
  const auto result = cudaGraphLaunch(slot.exec, slot.stream);
  return status(result, "cudaGraphLaunch failed");
}
CudaGraphStatus synchronize_inference_graph(std::size_t batch_size) noexcept {
  if (batch_size == 0 || batch_size > kMaxBatch || graphs[batch_size].exec == nullptr) return {CudaCapability::unavailable, "batch size was not captured at startup"};
  const auto& slot = graphs[batch_size];
  return status(cudaStreamSynchronize(slot.stream), "cudaStreamSynchronize failed");
}
CudaGraphStatus direct_inference_forward(std::size_t batch_size) noexcept {
  if (batch_size == 0 || batch_size > kMaxBatch || graphs[batch_size].device == nullptr) return {CudaCapability::unavailable, "batch size was not initialized at startup"};
  const auto& slot = graphs[batch_size];
  auto result = cudaMemcpyAsync(slot.device, slot.host, batch_size * sizeof(float), cudaMemcpyHostToDevice, slot.stream);
  if (result == cudaSuccess) { forward_stub<<<static_cast<unsigned>((batch_size + 255) / 256), 256, 0, slot.stream>>>(slot.device, batch_size); result = cudaGetLastError(); }
  if (result == cudaSuccess) result = cudaMemcpyAsync(slot.host, slot.device, batch_size * sizeof(float), cudaMemcpyDeviceToHost, slot.stream);
  if (result == cudaSuccess) result = cudaStreamSynchronize(slot.stream);
  return status(result, "direct CUDA forward failed");
}
void shutdown_inference_graphs() noexcept {
  std::lock_guard<std::mutex> lock(graph_mutex);
  for (auto& slot : graphs) { if (slot.exec) cudaGraphExecDestroy(slot.exec); if (slot.graph) cudaGraphDestroy(slot.graph); if (slot.device) cudaFree(slot.device); if (slot.host) cudaFreeHost(slot.host); if (slot.stream) cudaStreamDestroy(slot.stream); slot = {}; }
}
}  // namespace fraud::daemon
