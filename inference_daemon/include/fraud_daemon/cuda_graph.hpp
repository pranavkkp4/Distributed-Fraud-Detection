#pragma once

#include <cstddef>
#include <string_view>

namespace fraud::daemon {

enum class CudaCapability { unavailable, ready, error };
struct CudaGraphStatus final { CudaCapability capability; std::string_view message; };

// CUDA builds allocate pinned host/device buffers, capture H2D -> forward -> D2H
// once per supported batch size, instantiate cudaGraphExec_t, then replay it.
// CPU-only builds return unavailable without pretending CUDA Graph is active.
CudaGraphStatus capture_inference_graph(std::size_t batch_size) noexcept;
CudaGraphStatus launch_inference_graph(std::size_t batch_size) noexcept;
CudaGraphStatus synchronize_inference_graph(std::size_t batch_size) noexcept;
CudaGraphStatus replay_inference_graph(std::size_t batch_size) noexcept;
CudaGraphStatus direct_inference_forward(std::size_t batch_size) noexcept;
void shutdown_inference_graphs() noexcept;

}  // namespace fraud::daemon
