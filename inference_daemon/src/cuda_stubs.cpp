#include "fraud_daemon/cuda_graph.hpp"
#include "fraud_daemon/cutlass_int8_gemm.hpp"

namespace fraud::daemon {
#if !defined(FRAUD_DAEMON_CUDA_LINKED)
CudaGraphStatus capture_inference_graph(std::size_t) noexcept { return {CudaCapability::unavailable, "built without CUDA"}; }
CudaGraphStatus launch_inference_graph(std::size_t) noexcept { return {CudaCapability::unavailable, "built without CUDA"}; }
CudaGraphStatus synchronize_inference_graph(std::size_t) noexcept { return {CudaCapability::unavailable, "built without CUDA"}; }
CudaGraphStatus replay_inference_graph(std::size_t) noexcept { return {CudaCapability::unavailable, "built without CUDA"}; }
CudaGraphStatus direct_inference_forward(std::size_t) noexcept { return {CudaCapability::unavailable, "built without CUDA"}; }
void shutdown_inference_graphs() noexcept {}
#endif
#if !defined(FRAUD_DAEMON_HAVE_CUTLASS)
CutlassStatus cutlass_int8_gemm(const std::int8_t*, const std::int8_t*, std::int32_t*, int, int, int, void*) noexcept {
  return {false, "CUTLASS is not enabled; configure -DFRAUD_DAEMON_ENABLE_CUDA=ON -DFRAUD_DAEMON_ENABLE_CUTLASS=ON -DCUTLASS_PATH=..."};
}
#endif
}  // namespace fraud::daemon
