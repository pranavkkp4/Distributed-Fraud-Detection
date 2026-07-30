#include "fraud_engine/execution_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fraud_engine {
namespace {
bool bounded(float value, float limit = kMaxMicrokernelTensorMagnitude) {
  return std::isfinite(value) && std::fabs(value) <= limit;
}

Status validate_tensor_metadata(const E4M3TensorMetadata& metadata) {
  if (metadata.recommended_action != Fp8RecommendedAction::kClip &&
      metadata.recommended_action != Fp8RecommendedAction::kFallbackFp16)
    return {StatusCode::kInvalidArgument, "E4M3 recommended action is invalid"};
  if (!std::isfinite(metadata.quant_scale) || metadata.quant_scale <= 0.0F ||
      !std::isfinite(metadata.dequant_scale) || metadata.dequant_scale <= 0.0F ||
      !std::isfinite(metadata.clip_threshold) || metadata.clip_threshold <= 0.0F)
    return {StatusCode::kInvalidArgument,
            "E4M3 quant/dequant scales and clip threshold must be finite and positive"};
  const float inverse_product = metadata.quant_scale * metadata.dequant_scale;
  const float represented_threshold = metadata.dequant_scale * kFp8E4M3Max;
  const float threshold_tolerance =
      std::max(1.0e-5F, std::fabs(metadata.clip_threshold) * 1.0e-4F);
  if (std::fabs(inverse_product - 1.0F) > 1.0e-4F ||
      std::fabs(represented_threshold - metadata.clip_threshold) > threshold_tolerance)
    return {StatusCode::kInvalidArgument,
            "E4M3 quant/dequant scales and clip threshold are inconsistent"};
  return Status::Ok();
}

bool tensor_outlier(float value, const E4M3TensorMetadata& metadata) {
  return std::fabs(value) > metadata.clip_threshold &&
         metadata.recommended_action == Fp8RecommendedAction::kFallbackFp16;
}
}  // namespace

Status projection_gemm_fp32(const MicrokernelWeights& model,
                            const std::array<float, kFeatureWidth>& input,
                            std::array<float, kEmbeddingWidth>* output) {
  if (output == nullptr) return {StatusCode::kInvalidArgument, "projection output is null"};
  for (std::size_t row = 0; row < kEmbeddingWidth; ++row) {
    float accumulator = model.projection_bias[row];
    if (!bounded(accumulator))
      return {StatusCode::kOutOfRange, "projection bias exceeds the microkernel range"};
    for (std::size_t column = 0; column < kFeatureWidth; ++column) {
      const float weight = model.projection_weights[row * kFeatureWidth + column];
      if (!bounded(weight) || !bounded(input[column]))
        return {StatusCode::kOutOfRange, "projection tensor exceeds the microkernel range"};
      accumulator = std::fma(weight, input[column], accumulator);
      if (!bounded(accumulator, kMaxMicrokernelIntermediateMagnitude))
        return {StatusCode::kOutOfRange, "projection intermediate exceeds the microkernel range"};
    }
    (*output)[row] = accumulator;
  }
  return Status::Ok();
}

Status short_context_attention_fp32(
    const std::array<float, kEmbeddingWidth>& query,
    const std::array<float, kMaxContextLength * kEmbeddingWidth>& context,
    std::uint32_t context_length, std::array<float, kEmbeddingWidth>* output) {
  if (output == nullptr) return {StatusCode::kInvalidArgument, "attention output is null"};
  if (context_length > kMaxContextLength)
    return {StatusCode::kInvalidArgument, "context exceeds fixed limit"};
  output->fill(0.0F);
  for (float value : query)
    if (!bounded(value, kMaxMicrokernelIntermediateMagnitude))
      return {StatusCode::kOutOfRange, "attention query exceeds the microkernel range"};
  if (context_length == 0) return Status::Ok();

  const std::size_t first_token = kMaxContextLength - context_length;
  std::array<float, kMaxContextLength> logits{};
  const float inverse_scale = 1.0F / std::sqrt(static_cast<float>(kEmbeddingWidth));
  float max_logit = -std::numeric_limits<float>::infinity();
  for (std::size_t token = 0; token < context_length; ++token) {
    float dot = 0.0F;
    const std::size_t row = first_token + token;
    for (std::size_t channel = 0; channel < kEmbeddingWidth; ++channel) {
      const float key = context[row * kEmbeddingWidth + channel];
      if (!bounded(key))
        return {StatusCode::kOutOfRange, "active attention context exceeds the microkernel range"};
      dot = std::fma(query[channel], key, dot);
      if (!bounded(dot, kMaxMicrokernelIntermediateMagnitude))
        return {StatusCode::kOutOfRange, "attention dot product exceeds the microkernel range"};
    }
    logits[token] = dot * inverse_scale;
    if (!bounded(logits[token], kMaxMicrokernelIntermediateMagnitude))
      return {StatusCode::kOutOfRange, "attention logit exceeds the microkernel range"};
    max_logit = std::max(max_logit, logits[token]);
  }
  float denominator = 0.0F;
  for (std::size_t token = 0; token < context_length; ++token) {
    logits[token] = std::exp(logits[token] - max_logit);
    denominator += logits[token];
  }
  if (!std::isfinite(denominator) || denominator <= 0.0F)
    return {StatusCode::kOutOfRange, "attention softmax is non-finite"};
  for (std::size_t token = 0; token < context_length; ++token) {
    const float weight = logits[token] / denominator;
    const std::size_t row = first_token + token;
    for (std::size_t channel = 0; channel < kEmbeddingWidth; ++channel) {
      (*output)[channel] =
          std::fma(weight, context[row * kEmbeddingWidth + channel], (*output)[channel]);
      if (!bounded((*output)[channel], kMaxMicrokernelIntermediateMagnitude))
        return {StatusCode::kOutOfRange, "attention output exceeds the microkernel range"};
    }
  }
  return Status::Ok();
}

Status validate_fp8_eligibility(const MicrokernelWeights& weights,
                                const MicrokernelRequest& request) {
  if (request.context_length > kMaxContextLength)
    return {StatusCode::kInvalidArgument, "context exceeds fixed limit"};
  Status metadata_status = validate_tensor_metadata(request.feature_fp8);
  if (!metadata_status.ok()) return metadata_status;
  metadata_status = validate_tensor_metadata(request.context_fp8);
  if (!metadata_status.ok()) return metadata_status;
  metadata_status = validate_tensor_metadata(weights.weight_fp8);
  if (!metadata_status.ok()) return metadata_status;
  metadata_status = validate_tensor_metadata(weights.projection_fp8);
  if (!metadata_status.ok()) return metadata_status;
  if (weights.fp8_format == Fp8Format::kE5M2Unsupported)
    return {StatusCode::kUnsupported, "E5M2 is not supported by this microkernel"};
  if (weights.fp8_format != Fp8Format::kE4M3)
    return {StatusCode::kInvalidArgument, "FP8 format metadata is invalid"};
  if (!weights.weight_fp8.eligible_for_fp8 ||
      !weights.projection_fp8.eligible_for_fp8 ||
      !request.feature_fp8.eligible_for_fp8 ||
      !request.context_fp8.eligible_for_fp8)
    return {StatusCode::kUnsupported, "FP8 calibration did not mark every tensor eligible"};
  for (float value : request.transaction_features)
    if (tensor_outlier(value, request.feature_fp8))
      return {StatusCode::kUnsupported, "feature outlier exceeds calibrated E4M3 range"};
  for (float value : weights.projection_weights)
    if (tensor_outlier(value, weights.weight_fp8))
      return {StatusCode::kUnsupported, "weight outlier exceeds calibrated E4M3 range"};
  const std::size_t first_value =
      (kMaxContextLength - request.context_length) * kEmbeddingWidth;
  for (std::size_t i = first_value; i < request.context.size(); ++i)
    if (tensor_outlier(request.context[i], request.context_fp8))
      return {StatusCode::kUnsupported, "context outlier exceeds calibrated E4M3 range"};
  std::array<float, kEmbeddingWidth> projection{};
  const Status projection_status =
      projection_gemm_fp32(weights, request.transaction_features, &projection);
  if (!projection_status.ok()) return projection_status;
  for (float value : projection)
    if (tensor_outlier(value, weights.projection_fp8))
      return {StatusCode::kUnsupported, "projection outlier exceeds calibrated E4M3 range"};
  return Status::Ok();
}
}  // namespace fraud_engine
