#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace fraud_engine {

inline constexpr std::size_t kFeatureWidth = 32;
inline constexpr std::size_t kEmbeddingWidth = 64;
inline constexpr std::size_t kMaxContextLength = 16;
inline constexpr float kMaxMicrokernelTensorMagnitude = 32.0F;
inline constexpr float kMaxMicrokernelIntermediateMagnitude = 1.0e8F;
inline constexpr float kFp8E4M3Max = 448.0F;

enum class StatusCode : std::uint8_t {
  kOk = 0,
  kInvalidArgument,
  kOutOfRange,
  kOutOfMemory,
  kUnsupported,
  kUnavailable,
  kInternal,
};

struct Status {
  StatusCode code{StatusCode::kOk};
  const char* message{"ok"};
  [[nodiscard]] constexpr bool ok() const { return code == StatusCode::kOk; }
  static constexpr Status Ok() { return {}; }
};

// A Backend value on a MicrokernelResult always describes the implementation
// that produced that result. kNotExecuted is used before any inference occurs.
enum class Backend : std::uint8_t { kNotExecuted, kCpuReference, kCudaFp16, kCudaFp8 };

enum class FallbackReason : std::uint8_t {
  kNone,
  kUnsupportedDevice,
  kFp8Ineligible,
  kUnsupportedFp8Format,
  kCudaUnavailable,
  kCudaOutOfMemory,
};

enum class Fp8Format : std::uint8_t { kE4M3, kE5M2Unsupported };
enum class Fp8RecommendedAction : std::uint8_t { kClip, kFallbackFp16 };

struct E4M3TensorMetadata {
  // quantized = clip(value * quant_scale, -448, 448)
  // dequantized = quantized * dequant_scale
  float quant_scale{1.0F};
  float dequant_scale{1.0F};
  float clip_threshold{kFp8E4M3Max};
  bool eligible_for_fp8{false};
  Fp8RecommendedAction recommended_action{Fp8RecommendedAction::kFallbackFp16};
};

struct DeviceCapabilities {
  bool cuda_compiled{false};
  bool cuda_available{false};
  int compute_capability_major{0};
  int compute_capability_minor{0};
};

// This chooses a preferred precision from capabilities; it does not claim that a
// request has executed. FP8 storage/conversion requires SM 8.9+. Runtime dispatch
// may fall back and reports that separately. The CUDA kernels use scalar low-
// precision operands with FP32 accumulation and make no Tensor Core claim.
[[nodiscard]] DeviceCapabilities detect_device_capabilities();
[[nodiscard]] Backend select_backend(const DeviceCapabilities& capabilities);
[[nodiscard]] const char* backend_name(Backend backend);

// This API is a projection-plus-single-head-attention microkernel pipeline used
// for accuracy checks and profiling. It cannot load or execute the exported full
// fraud transformer, and its score must not be presented as full-model parity.
struct MicrokernelRequest {
  std::array<float, kFeatureWidth> transaction_features{};
  // Chronological history, left-padded with zero rows when context_length is short.
  std::array<float, kMaxContextLength * kEmbeddingWidth> context{};
  std::uint32_t context_length{0};
  E4M3TensorMetadata feature_fp8{};
  E4M3TensorMetadata context_fp8{};
};

struct MicrokernelWeights {
  // Row-major: [embedding_width, feature_width]. All accumulation remains FP32.
  std::array<float, kEmbeddingWidth * kFeatureWidth> projection_weights{};
  std::array<float, kEmbeddingWidth> projection_bias{};
  E4M3TensorMetadata weight_fp8{};
  E4M3TensorMetadata projection_fp8{};
  Fp8Format fp8_format{Fp8Format::kE4M3};
};

struct MicrokernelResult {
  Backend preferred_backend{Backend::kCpuReference};
  Backend backend{Backend::kNotExecuted};
  bool fallback_used{false};
  bool eligible_for_fp8{false};
  FallbackReason fallback_reason{FallbackReason::kNone};
  Status fallback_status{};
  FallbackReason terminal_fallback_reason{FallbackReason::kNone};
  Status terminal_fallback_status{};
  std::array<float, kEmbeddingWidth> projection{};
  std::array<float, kEmbeddingWidth> attended_context{};
  float anomaly_score{0.0F};
};

// Memory returned by this arena belongs to the arena and remains valid until reset
// or destruction. CUDA builds use stream-ordered allocation when a CUDA device is
// available; the portable implementation is used otherwise. An arena is bound to
// the CUDA device current at construction. Instances are not thread-safe; callers
// must externally serialize allocate/reset/synchronize operations.
class StreamOrderedArena {
 public:
  explicit StreamOrderedArena(std::size_t initial_bytes = 0);
  ~StreamOrderedArena();
  StreamOrderedArena(StreamOrderedArena&&) noexcept;
  StreamOrderedArena& operator=(StreamOrderedArena&&) noexcept;
  StreamOrderedArena(const StreamOrderedArena&) = delete;
  StreamOrderedArena& operator=(const StreamOrderedArena&) = delete;

  [[nodiscard]] Status allocate(std::size_t bytes, std::size_t alignment, void** result);
  // Surfaces asynchronous CUDA errors. CPU-backed arenas return success.
  [[nodiscard]] Status synchronize();
  [[nodiscard]] Status reset() noexcept;
  [[nodiscard]] std::size_t bytes_in_use() const noexcept;
  [[nodiscard]] bool is_cuda_backed() const noexcept;
  [[nodiscard]] int device_ordinal() const noexcept;
  // Returns cudaStream_t as an opaque pointer for CUDA-backed arenas, else null.
  // CUDA work consuming an allocation must use this stream or establish an
  // explicit stream dependency before accessing the allocation elsewhere.
  [[nodiscard]] void* native_stream() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// CPU graph replay preserves the lifecycle and fixed-shape contract used by CUDA
// Graphs without pretending that a CPU lambda is a captured GPU graph.
class GraphReplay {
 public:
  using Work = std::function<Status()>;
  [[nodiscard]] Status capture(Work work);
  // CUDA Graph capture is deliberately reported unavailable until a captured
  // fixed-buffer CUDA dispatch implementation exists.
  [[nodiscard]] Status capture_cuda();
  [[nodiscard]] Status replay() const;
  [[nodiscard]] bool captured() const noexcept;
  void reset() noexcept;

 private:
  Work work_;
};

class MicrokernelPipeline {
 public:
  explicit MicrokernelPipeline(DeviceCapabilities capabilities = detect_device_capabilities());
  // Last backend that successfully produced a result, initially kNotExecuted.
  [[nodiscard]] Backend backend() const noexcept;
  [[nodiscard]] Backend preferred_backend() const noexcept;
  [[nodiscard]] Status run(const MicrokernelWeights& weights, const MicrokernelRequest& request,
                           MicrokernelResult* result) const;

 private:
  Backend preferred_backend_;
  mutable std::atomic<Backend> last_executed_backend_{Backend::kNotExecuted};
};

// Independently callable FP32-reference primitives, useful for accuracy testing.
[[nodiscard]] Status projection_gemm_fp32(const MicrokernelWeights& model,
                                          const std::array<float, kFeatureWidth>& input,
                                          std::array<float, kEmbeddingWidth>* output);
[[nodiscard]] Status short_context_attention_fp32(
    const std::array<float, kEmbeddingWidth>& query,
    const std::array<float, kMaxContextLength * kEmbeddingWidth>& context,
    std::uint32_t context_length, std::array<float, kEmbeddingWidth>* output);

// Returns whether every E4M3 tensor can be represented using the supplied
// per-tensor scales, including the projected query intermediate.
[[nodiscard]] Status validate_fp8_eligibility(const MicrokernelWeights& weights,
                                              const MicrokernelRequest& request);

}  // namespace fraud_engine
