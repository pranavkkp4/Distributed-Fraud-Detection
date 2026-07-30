#include "fraud_daemon/cutlass_int8_gemm.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>

namespace fraud::daemon {
CutlassStatus cutlass_int8_gemm(const std::int8_t* a, const std::int8_t* b, std::int32_t* c,
                                int m, int n, int k, void* stream) noexcept {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0 || (k % 32) != 0) {
    return {false, "invalid INT8 GEMM arguments (K must be a multiple of 32)"};
  }
  using Int8Gemm = cutlass::gemm::device::Gemm<
      std::int8_t, cutlass::layout::RowMajor,
      std::int8_t, cutlass::layout::ColumnMajor,
      std::int32_t, cutlass::layout::RowMajor,
      std::int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 32>>;
  Int8Gemm gemm;
  typename Int8Gemm::Arguments arguments({m, n, k}, {a, k}, {b, k}, {c, n}, {c, n}, {1, 0});
  const auto result = gemm(arguments, nullptr, reinterpret_cast<cudaStream_t>(stream));
  return result == cutlass::Status::kSuccess ? CutlassStatus{true, "ready"} : CutlassStatus{false, "CUTLASS GEMM launch failed"};
}
}  // namespace fraud::daemon
