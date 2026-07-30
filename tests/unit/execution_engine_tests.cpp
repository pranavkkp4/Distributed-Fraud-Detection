#include "fraud_engine/execution_engine.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

using namespace fraud_engine;
namespace {
int failures = 0;
void expect(bool condition, const char* message) { if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; } }
bool near(float left, float right, float tolerance = 1.0e-5F) { return std::fabs(left - right) <= tolerance; }

E4M3TensorMetadata fp8_metadata(
    float dequant_scale,
    Fp8RecommendedAction action = Fp8RecommendedAction::kFallbackFp16) {
  E4M3TensorMetadata metadata{};
  metadata.quant_scale = 1.0F / dequant_scale;
  metadata.dequant_scale = dequant_scale;
  metadata.clip_threshold = kFp8E4M3Max * dequant_scale;
  metadata.eligible_for_fp8 = true;
  metadata.recommended_action = action;
  return metadata;
}

MicrokernelWeights model() {
  MicrokernelWeights value{};
  for (std::size_t row = 0; row < kEmbeddingWidth; ++row) {
    value.projection_bias[row] = static_cast<float>(row) * 0.01F;
    for (std::size_t column = 0; column < kFeatureWidth; ++column)
      value.projection_weights[row * kFeatureWidth + column] = static_cast<float>((row + 1) * (column + 1)) * 0.001F;
  }
  return value;
}
void test_projection() {
  std::array<float, kFeatureWidth> input{}; input.fill(1.0F);
  std::array<float, kEmbeddingWidth> result{};
  expect(projection_gemm_fp32(model(), input, &result).ok(), "projection succeeds");
  expect(near(result[0], 0.528F), "projection uses FP32 accumulation and bias");
  expect(near(result[15], 8.598F), "projection calculates final row");
  expect(near(result[63], 34.422F), "projection matches the 64-channel model contract");
}
void test_attention() {
  std::array<float, kEmbeddingWidth> query{}; query[0] = 4.0F;
  std::array<float, kMaxContextLength * kEmbeddingWidth> context{};
  const std::size_t suffix = (kMaxContextLength - 2) * kEmbeddingWidth;
  context[0] = 31.0F;  // Inactive left padding must not participate.
  context[suffix] = 1.0F;
  context[suffix + kEmbeddingWidth] = 3.0F;
  std::array<float, kEmbeddingWidth> output{};
  expect(short_context_attention_fp32(query, context, 2, &output).ok(), "attention succeeds");
  expect(output[0] > 2.4F && output[0] < 3.01F, "attention favors matching context");
  expect(short_context_attention_fp32(query, context, 17, &output).code == StatusCode::kInvalidArgument,
         "attention rejects oversized context");
}
void test_lifecycle() {
  StreamOrderedArena arena;
  void* first = nullptr; void* second = nullptr;
  expect(arena.allocate(64, 16, &first).ok() && first != nullptr, "arena allocates aligned storage");
  expect(arena.allocate(32, 32, &second).ok() && second != nullptr, "arena supports multiple allocations");
  expect(reinterpret_cast<std::uintptr_t>(second) % 32 == 0, "arena honors requested alignment");
  expect(!arena.is_cuda_backed() || arena.native_stream() != nullptr,
         "CUDA arena exposes the stream that orders its allocations");
  expect(arena.is_cuda_backed() ? arena.device_ordinal() >= 0 : arena.device_ordinal() == -1,
         "arena reports explicit CPU or owning CUDA device");
  expect(arena.synchronize().ok(), "arena synchronization surfaces no asynchronous errors");
  expect(arena.bytes_in_use() == 96, "arena tracks usage");
  expect(arena.reset().ok(), "arena reset synchronizes pending frees");
  expect(arena.bytes_in_use() == 0, "arena reset releases allocations");
  GraphReplay graph; int launches = 0;
  expect(graph.replay().code == StatusCode::kUnavailable, "uncaptured graph is unavailable");
  expect(graph.capture([&launches] { ++launches; return Status::Ok(); }).ok(), "graph capture works");
  expect(graph.replay().ok() && graph.replay().ok() && launches == 2, "graph replay is repeatable");
  expect(graph.capture_cuda().code == StatusCode::kUnavailable,
         "CUDA Graph capture is explicitly unavailable, not simulated");
}
void test_engine_contract() {
  MicrokernelRequest request{};
  request.context_length = 1;
  request.context[(kMaxContextLength - 1) * kEmbeddingWidth] = 1.0F;
  for (std::size_t i = 0; i < kFeatureWidth; ++i) request.transaction_features[i] = 0.5F;
  MicrokernelPipeline pipeline(DeviceCapabilities{});
  MicrokernelResult result{};
  expect(pipeline.backend() == Backend::kNotExecuted, "pipeline reports no execution before run");
  expect(pipeline.preferred_backend() == Backend::kCpuReference, "pipeline exposes CPU preference");
  expect(pipeline.run(model(), request, &result).ok(), "fixed-shape microkernel succeeds");
  expect(result.backend == Backend::kCpuReference, "CPU backend reports actual execution");
  expect(result.preferred_backend == Backend::kCpuReference && !result.fallback_used,
         "CPU execution is not labeled as a fallback");
  expect(pipeline.backend() == result.backend, "pipeline records last executed backend");
  std::array<float, kEmbeddingWidth> reference{};
  expect(projection_gemm_fp32(model(), request.transaction_features, &reference).ok(), "reference projection succeeds");
  for (std::size_t i = 0; i < kEmbeddingWidth; ++i)
    expect(near(result.projection[i], reference[i]), "engine projection matches FP32 reference");
  expect(result.anomaly_score >= 0.0F, "score is non-negative");
  expect(select_backend({true, true, 8, 6}) == Backend::kCudaFp16, "pre-FP8 GPU selects FP16");
  expect(select_backend({true, true, 8, 9}) == Backend::kCudaFp8, "supported GPU selects FP8");
  expect(select_backend({true, true, 7, 0}) == Backend::kCpuReference,
         "device below the compiled architecture floor selects CPU");
}

void test_range_rejection() {
  MicrokernelRequest request{};
  request.transaction_features[0] = std::numeric_limits<float>::max();
  MicrokernelPipeline pipeline(DeviceCapabilities{});
  MicrokernelResult result{};
  expect(pipeline.run(model(), request, &result).code == StatusCode::kOutOfRange,
         "extreme finite tensor is rejected before execution");
  expect(pipeline.backend() == Backend::kNotExecuted && result.backend == Backend::kNotExecuted,
         "rejected input is not labeled as executed");

  request = {};
  request.feature_fp8.quant_scale = 0.0F;
  expect(pipeline.run(model(), request, &result).code == StatusCode::kInvalidArgument,
         "invalid FP8 scale is rejected even on the CPU path");
}

void test_fp8_outlier_fallback() {
  MicrokernelRequest request{};
  request.context_length = 1;
  request.feature_fp8 = fp8_metadata(0.0001F);
  request.context_fp8 = fp8_metadata(0.01F);
  request.transaction_features.fill(0.5F);
  request.context[(kMaxContextLength - 1) * kEmbeddingWidth] = 0.25F;
  MicrokernelWeights calibrated_model = model();
  calibrated_model.weight_fp8 = fp8_metadata(0.005F);
  calibrated_model.projection_fp8 = fp8_metadata(0.02F);

  MicrokernelPipeline pipeline({true, true, 8, 9});
  MicrokernelResult result{};
  expect(pipeline.run(calibrated_model, request, &result).ok(),
         "FP8 activation outlier takes a recoverable fallback");
  expect(result.preferred_backend == Backend::kCudaFp8 && result.fallback_used,
         "outlier preserves the original FP8 preference");
  expect(!result.eligible_for_fp8, "outlier clears explicit FP8 eligibility");
  expect(result.fallback_reason == FallbackReason::kFp8Ineligible &&
             result.fallback_status.code == StatusCode::kUnsupported,
         "outlier fallback retains structured classification");
  expect(result.terminal_fallback_reason != FallbackReason::kNone &&
             result.terminal_fallback_status.code != StatusCode::kOk,
         "fallback retains the terminal classified event as well");
  expect(std::string(result.fallback_status.message).find("outlier") != std::string::npos,
         "fallback status retains a stable diagnostic message");
  const DeviceCapabilities capabilities = detect_device_capabilities();
  const Backend expected = capabilities.cuda_available &&
                                   (capabilities.compute_capability_major * 10 +
                                    capabilities.compute_capability_minor) >= 75
      ? Backend::kCudaFp16
      : Backend::kCpuReference;
  expect(result.backend == expected, "outlier reports the FP16 or CPU backend actually executed");

  MicrokernelRequest clipping_request = request;
  clipping_request.feature_fp8 =
      fp8_metadata(0.0001F, Fp8RecommendedAction::kClip);
  MicrokernelWeights clipping_model = calibrated_model;
  clipping_model.projection_fp8 = fp8_metadata(0.1F);
  const Status clipping_status = validate_fp8_eligibility(clipping_model, clipping_request);
  if (!clipping_status.ok()) std::cerr << "clip eligibility: " << clipping_status.message << '\n';
  expect(clipping_status.ok(),
         "explicit clipping recommendation permits a calibrated outlier");

  calibrated_model.fp8_format = Fp8Format::kE5M2Unsupported;
  request.feature_fp8 = fp8_metadata(0.01F);
  MicrokernelResult e5m2{};
  expect(pipeline.run(calibrated_model, request, &e5m2).ok(),
         "unsupported E5M2 takes a recoverable lower-precision path");
  expect(e5m2.fallback_reason == FallbackReason::kUnsupportedFp8Format,
         "E5M2 is explicitly classified as unsupported");
}

void test_runtime_dispatch_reporting() {
  MicrokernelRequest request{};
  request.context_length = 2;
  request.feature_fp8 = fp8_metadata(0.01F);
  request.context_fp8 = fp8_metadata(0.02F);
  for (std::size_t i = 0; i < kFeatureWidth; ++i)
    request.transaction_features[i] = static_cast<float>(i + 1) * 0.01F;
  const std::size_t suffix = (kMaxContextLength - request.context_length) * kEmbeddingWidth;
  for (std::size_t i = 0; i < 2 * kEmbeddingWidth; ++i)
    request.context[suffix + i] = static_cast<float>((i % kEmbeddingWidth) + 1) * 0.02F;

  const DeviceCapabilities capabilities = detect_device_capabilities();
  MicrokernelWeights calibrated_model = model();
  calibrated_model.weight_fp8 = fp8_metadata(0.005F);
  calibrated_model.projection_fp8 = fp8_metadata(0.02F);
  MicrokernelPipeline pipeline(capabilities);
  MicrokernelResult result{};
  expect(pipeline.run(calibrated_model, request, &result).ok(), "runtime-selected microkernel succeeds");
  expect(result.preferred_backend == select_backend(capabilities),
         "result preserves the capability-derived preference");
  expect(result.eligible_for_fp8, "consistent calibrated metadata is FP8 eligible");
  expect(result.backend == pipeline.backend(), "result and last actual backend agree");
  if (!capabilities.cuda_available) {
    expect(result.backend == Backend::kCpuReference, "unavailable CUDA executes on CPU");
  } else {
    expect(result.backend == result.preferred_backend,
           "supported CUDA preference is dispatched instead of merely relabeled");
    expect(!result.fallback_used, "successful CUDA dispatch is not a fallback");
  }

  std::array<float, kEmbeddingWidth> reference{};
  expect(projection_gemm_fp32(calibrated_model, request.transaction_features, &reference).ok(),
         "dispatch reference projection succeeds");
  const float tolerance = result.backend == Backend::kCudaFp8 ? 0.6F : 0.01F;
  for (std::size_t i = 0; i < kEmbeddingWidth; ++i)
    expect(near(result.projection[i], reference[i], tolerance),
           "executed backend stays within its precision tolerance");
  std::array<float, kEmbeddingWidth> attention_reference{};
  expect(short_context_attention_fp32(reference, request.context, request.context_length,
                                      &attention_reference).ok(),
         "dispatch reference attention succeeds");
  for (std::size_t i = 0; i < kEmbeddingWidth; ++i)
    expect(near(result.attended_context[i], attention_reference[i], tolerance),
           "executed attention stays within its precision tolerance");
  expect(std::isfinite(result.anomaly_score), "executed backend produces a finite score");

  MicrokernelPipeline requested_fp8({true, true, 8, 9});
  MicrokernelResult fallback{};
  expect(requested_fp8.run(calibrated_model, request, &fallback).ok(),
         "unsupported preference falls back safely");
  if (!capabilities.cuda_available || capabilities.compute_capability_major < 8 ||
      (capabilities.compute_capability_major == 8 && capabilities.compute_capability_minor < 9)) {
    expect(fallback.preferred_backend == Backend::kCudaFp8, "FP8 remains a preference only");
    const Backend expected_fallback = capabilities.cuda_available
        ? Backend::kCudaFp16
        : Backend::kCpuReference;
    expect(fallback.backend == expected_fallback && fallback.fallback_used,
           "FP8 request is labeled with the lower-precision fallback actually executed");
    const FallbackReason expected_reason = capabilities.cuda_available
        ? FallbackReason::kUnsupportedDevice
        : FallbackReason::kCudaUnavailable;
    expect(fallback.fallback_reason == expected_reason &&
               (fallback.fallback_status.code == StatusCode::kUnsupported ||
                fallback.fallback_status.code == StatusCode::kUnavailable),
           "unsupported runtime fallback retains a structured stable classification");
  }
}
}
int main() {
  test_projection();
  test_attention();
  test_lifecycle();
  test_engine_contract();
  test_range_rejection();
  test_fp8_outlier_fallback();
  test_runtime_dispatch_reporting();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
