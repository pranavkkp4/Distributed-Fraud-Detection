#include "fraud_engine/execution_engine.hpp"

#include <utility>

namespace fraud_engine {
Status GraphReplay::capture(Work work) {
  if (!work) return {StatusCode::kInvalidArgument, "graph work is empty"};
  work_ = std::move(work);
  return Status::Ok();
}
Status GraphReplay::capture_cuda() {
  return {StatusCode::kUnavailable, "CUDA Graph capture is not implemented"};
}
Status GraphReplay::replay() const {
  if (!work_) return {StatusCode::kUnavailable, "graph has not been captured"};
  return work_();
}
bool GraphReplay::captured() const noexcept { return static_cast<bool>(work_); }
void GraphReplay::reset() noexcept { work_ = {}; }
}  // namespace fraud_engine
