#include "fraud_daemon/batcher.hpp"

#include <chrono>

namespace fraud::daemon {
std::uint64_t monotonic_time_ns() noexcept {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace fraud::daemon
