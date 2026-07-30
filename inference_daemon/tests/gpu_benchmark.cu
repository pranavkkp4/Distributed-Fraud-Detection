#include "fraud_daemon/cuda_graph.hpp"
#include "fraud_daemon/cutlass_int8_gemm.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
bool check_cuda(cudaError_t result, const char* action) {
  if (result == cudaSuccess) return true;
  std::cerr << action << ": " << cudaGetErrorString(result) << '\n'; return false;
}
double elapsed_us(bool graph, int iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const auto result = graph ? fraud::daemon::replay_inference_graph(32)
                              : fraud::daemon::direct_inference_forward(32);
    if (result.capability != fraud::daemon::CudaCapability::ready) return -1;
  }
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(stop - start).count() / iterations;
}
}

int main() {
  constexpr int m = 128, n = 128, k = 64;
  std::vector<std::int8_t> a(m * k), b(k * n);
  std::vector<std::int32_t> expected(m * n, 0), actual(m * n, 0);
  for (int i = 0; i < m * k; ++i) a[i] = static_cast<std::int8_t>((i % 7) - 3);
  for (int i = 0; i < k * n; ++i) b[i] = static_cast<std::int8_t>((i % 5) - 2);
  // B is column-major as configured in cutlass_int8_gemm.
  for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col)
    for (int inner = 0; inner < k; ++inner) expected[row * n + col] += static_cast<int>(a[row * k + inner]) * b[col * k + inner];
  std::int8_t* device_a{}; std::int8_t* device_b{}; std::int32_t* device_c{};
  if (!check_cuda(cudaMalloc(&device_a, a.size()), "cudaMalloc A") || !check_cuda(cudaMalloc(&device_b, b.size()), "cudaMalloc B") || !check_cuda(cudaMalloc(&device_c, actual.size() * sizeof(std::int32_t)), "cudaMalloc C")) return 1;
  if (!check_cuda(cudaMemcpy(device_a, a.data(), a.size(), cudaMemcpyHostToDevice), "copy A") || !check_cuda(cudaMemcpy(device_b, b.data(), b.size(), cudaMemcpyHostToDevice), "copy B")) return 1;
  const auto gemm = fraud::daemon::cutlass_int8_gemm(device_a, device_b, device_c, m, n, k, nullptr);
  if (!gemm.available) { std::cerr << gemm.message << '\n'; return 2; }
  if (!check_cuda(cudaDeviceSynchronize(), "CUTLASS sync") || !check_cuda(cudaMemcpy(actual.data(), device_c, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "copy C")) return 1;
  for (std::size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) { std::cerr << "INT8 GEMM mismatch at " << i << "\n"; return 1; }
  if (fraud::daemon::capture_inference_graph(32).capability != fraud::daemon::CudaCapability::ready) return 1;
  const double graph_us = elapsed_us(true, 500);
  const double direct_us = elapsed_us(false, 500);
  fraud::daemon::shutdown_inference_graphs();
  cudaFree(device_c); cudaFree(device_b); cudaFree(device_a);
  if (graph_us < 0 || direct_us < 0) return 1;
  std::cout << "{\"gpu\":\"cuda\",\"m\":128,\"n\":128,\"k\":64,\"int8_gemm\":\"pass\",\"graph_replay_us\":" << graph_us << ",\"direct_us\":" << direct_us << "}\n";
  return 0;
}
