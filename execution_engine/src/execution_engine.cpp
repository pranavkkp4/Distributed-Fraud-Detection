#include "fraud_engine/execution_engine.hpp"

#include <cmath>
#include <cstddef>

namespace fraud_engine {
#ifdef FRAUD_ENGINE_WITH_CUDA
DeviceCapabilities cuda_device_capabilities();
Status cuda_run_microkernel(const MicrokernelWeights& weights,
                            const MicrokernelRequest& request, Backend backend,
                            MicrokernelResult* result);
#endif

namespace {
#ifndef FRAUD_ENGINE_MIN_CUDA_CC
#define FRAUD_ENGINE_MIN_CUDA_CC 75
#endif
#ifndef FRAUD_ENGINE_FP8_MIN_CUDA_CC
#define FRAUD_ENGINE_FP8_MIN_CUDA_CC 89
#endif
constexpr int kMinimumCompiledComputeCapability = FRAUD_ENGINE_MIN_CUDA_CC;

bool recoverable_cuda_failure(const Status& status) {
  return status.code == StatusCode::kUnsupported ||
         status.code == StatusCode::kUnavailable ||
         status.code == StatusCode::kOutOfMemory;
}

FallbackReason fallback_reason_for(const Status& status) {
  switch (status.code) {
    case StatusCode::kUnsupported: return FallbackReason::kUnsupportedDevice;
    case StatusCode::kUnavailable: return FallbackReason::kCudaUnavailable;
    case StatusCode::kOutOfMemory: return FallbackReason::kCudaOutOfMemory;
    default: return FallbackReason::kNone;
  }
}

void record_fallback(MicrokernelResult* result, const Status& status, FallbackReason reason) {
  if (!result->fallback_used) {
    result->fallback_status = status;
    result->fallback_reason = reason;
  }
  result->fallback_used = true;
  result->terminal_fallback_status = status;
  result->terminal_fallback_reason = reason;
}

Status finalize_score(MicrokernelResult* result) {
  float magnitude_squared = 0.0F;
  for (std::size_t i = 0; i < kEmbeddingWidth; ++i) {
    if (!std::isfinite(result->projection[i]) ||
        !std::isfinite(result->attended_context[i]) ||
        std::fabs(result->projection[i]) > kMaxMicrokernelIntermediateMagnitude ||
        std::fabs(result->attended_context[i]) > kMaxMicrokernelIntermediateMagnitude)
      return {StatusCode::kInternal, "executed backend produced an invalid output"};
    const float difference = result->projection[i] - result->attended_context[i];
    magnitude_squared = std::fma(difference, difference, magnitude_squared);
    if (!std::isfinite(magnitude_squared))
      return {StatusCode::kInternal, "executed backend produced an invalid score intermediate"};
  }
  result->anomaly_score =
      std::sqrt(magnitude_squared / static_cast<float>(kEmbeddingWidth));
  if (!std::isfinite(result->anomaly_score))
    return {StatusCode::kInternal, "executed backend produced a non-finite score"};
  return Status::Ok();
}
}  // namespace

DeviceCapabilities detect_device_capabilities() {
#ifdef FRAUD_ENGINE_WITH_CUDA
  return cuda_device_capabilities();
#else
  return {};
#endif
}

Backend select_backend(const DeviceCapabilities& capabilities) {
  if (!capabilities.cuda_compiled || !capabilities.cuda_available)
    return Backend::kCpuReference;
  const int compute_capability =
      capabilities.compute_capability_major * 10 + capabilities.compute_capability_minor;
  if (compute_capability < kMinimumCompiledComputeCapability)
    return Backend::kCpuReference;
  if (compute_capability >= FRAUD_ENGINE_FP8_MIN_CUDA_CC) return Backend::kCudaFp8;
  return Backend::kCudaFp16;
}

const char* backend_name(Backend backend) {
  switch (backend) {
    case Backend::kNotExecuted: return "not-executed";
    case Backend::kCpuReference: return "cpu-reference-microkernel";
    case Backend::kCudaFp16: return "cuda-fp16-microkernel";
    case Backend::kCudaFp8: return "cuda-e4m3-microkernel";
  }
  return "unknown";
}

MicrokernelPipeline::MicrokernelPipeline(DeviceCapabilities capabilities)
    : preferred_backend_(select_backend(capabilities)) {}

Backend MicrokernelPipeline::backend() const noexcept {
  return last_executed_backend_.load(std::memory_order_relaxed);
}

Backend MicrokernelPipeline::preferred_backend() const noexcept { return preferred_backend_; }

Status MicrokernelPipeline::run(const MicrokernelWeights& weights,
                                const MicrokernelRequest& request,
                                MicrokernelResult* result) const {
  if (result == nullptr) return {StatusCode::kInvalidArgument, "microkernel result is null"};
  if (request.context_length > kMaxContextLength)
    return {StatusCode::kInvalidArgument, "context exceeds fixed limit"};
  // The reference pass is the contract validator and the recoverable fallback.
  // It is intentionally part of this correctness microbenchmark, not a latency claim.
  std::array<float, kEmbeddingWidth> reference_projection{};
  Status status = projection_gemm_fp32(weights, request.transaction_features,
                                       &reference_projection);
  if (!status.ok()) return status;
  std::array<float, kEmbeddingWidth> reference_attention{};
  status = short_context_attention_fp32(reference_projection, request.context,
                                        request.context_length, &reference_attention);
  if (!status.ok()) return status;

  MicrokernelResult next{};
  next.preferred_backend = preferred_backend_;
  Backend candidate = preferred_backend_;
  const Status fp8_eligibility = validate_fp8_eligibility(weights, request);
  if (fp8_eligibility.code == StatusCode::kInvalidArgument) return fp8_eligibility;
  next.eligible_for_fp8 = fp8_eligibility.ok();
  if (candidate == Backend::kCudaFp8 && !fp8_eligibility.ok()) {
    if (!recoverable_cuda_failure(fp8_eligibility)) return fp8_eligibility;
    const FallbackReason reason = weights.fp8_format == Fp8Format::kE5M2Unsupported
        ? FallbackReason::kUnsupportedFp8Format
        : FallbackReason::kFp8Ineligible;
    record_fallback(&next, fp8_eligibility, reason);
    candidate = Backend::kCudaFp16;
  }

#ifdef FRAUD_ENGINE_WITH_CUDA
  if (preferred_backend_ == Backend::kCudaFp8 ||
      preferred_backend_ == Backend::kCudaFp16) {
    status = cuda_run_microkernel(weights, request, candidate, &next);
    if (status.ok()) {
      next.backend = candidate;
      next.fallback_used = next.fallback_used || candidate != preferred_backend_;
      const Status final_status = finalize_score(&next);
      if (!final_status.ok()) return final_status;
      *result = next;
      last_executed_backend_.store(next.backend, std::memory_order_relaxed);
      return Status::Ok();
    }
    if (!recoverable_cuda_failure(status)) return status;

    // A capability object supplied by a caller can request FP8 on a non-FP8
    // current device. An unsupported FP8 launch gets one honest FP16 attempt.
    if (candidate == Backend::kCudaFp8 && status.code == StatusCode::kUnsupported) {
      record_fallback(&next, status, FallbackReason::kUnsupportedDevice);
      status = cuda_run_microkernel(weights, request, Backend::kCudaFp16, &next);
      if (status.ok()) {
        next.backend = Backend::kCudaFp16;
        const Status final_status = finalize_score(&next);
        if (!final_status.ok()) return final_status;
        *result = next;
        last_executed_backend_.store(next.backend, std::memory_order_relaxed);
        return Status::Ok();
      }
      if (!recoverable_cuda_failure(status)) return status;
    }
    record_fallback(&next, status, fallback_reason_for(status));
  }
#else
  if (preferred_backend_ != Backend::kCpuReference) {
    const Status unavailable{StatusCode::kUnavailable,
                             "CUDA support was not compiled into this microkernel"};
    record_fallback(&next, unavailable, FallbackReason::kCudaUnavailable);
  }
#endif

  next.projection = reference_projection;
  next.attended_context = reference_attention;
  next.backend = Backend::kCpuReference;
  status = finalize_score(&next);
  if (!status.ok()) return status;
  *result = next;
  last_executed_backend_.store(next.backend, std::memory_order_relaxed);
  return Status::Ok();
}
}  // namespace fraud_engine
