#include "fraud_daemon/cuda_graph.hpp"
#include "fraud_daemon/cutlass_int8_gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
bool check_cuda(cudaError_t result, const char* action) {
  if (result == cudaSuccess) return true;
  std::cerr << action << ": " << cudaGetErrorString(result) << '\n'; return false;
}
struct LatencySummary {
  double mean{};
  double p50{};
  double p90{};
  double p99{};
  double p999{};
  double minimum{};
  double maximum{};
  bool valid{};
};

LatencySummary elapsed_us(bool graph, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    const auto result = graph ? fraud::daemon::replay_inference_graph(32)
                              : fraud::daemon::direct_inference_forward(32);
    if (result.capability != fraud::daemon::CudaCapability::ready) return {};
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = graph ? fraud::daemon::replay_inference_graph(32)
                              : fraud::daemon::direct_inference_forward(32);
    const auto stop = std::chrono::steady_clock::now();
    if (result.capability != fraud::daemon::CudaCapability::ready) return {};
    samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
  }
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double quantile) {
    const auto rank = static_cast<std::size_t>(std::ceil(quantile * samples.size()));
    return samples[(std::max<std::size_t>)(1, rank) - 1];
  };
  return {std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
          percentile(0.50), percentile(0.90), percentile(0.99), percentile(0.999),
          samples.front(), samples.back(), true};
}

void print_summary(const char* name, const LatencySummary& value) {
  std::cout << '"' << name << "\":{\"mean\":" << value.mean << ",\"p50\":" << value.p50
            << ",\"p90\":" << value.p90 << ",\"p99\":" << value.p99
            << ",\"p999\":" << value.p999 << ",\"min\":" << value.minimum
            << ",\"max\":" << value.maximum << '}';
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
  constexpr int warmup = 100, iterations = 2000;
  const auto graph_us = elapsed_us(true, warmup, iterations);
  const auto direct_us = elapsed_us(false, warmup, iterations);
  fraud::daemon::shutdown_inference_graphs();
  cudaFree(device_c); cudaFree(device_b); cudaFree(device_a);
  if (!graph_us.valid || !direct_us.valid) return 1;
  std::cout << "{\"gpu\":\"cuda\",\"m\":128,\"n\":128,\"k\":64,\"int8_gemm\":\"pass\",\"warmup\":"
            << warmup << ",\"iterations\":" << iterations << ',';
  print_summary("graph_wall_us", graph_us); std::cout << ',';
  print_summary("direct_wall_us", direct_us); std::cout << "}\n";
  return 0;
}
