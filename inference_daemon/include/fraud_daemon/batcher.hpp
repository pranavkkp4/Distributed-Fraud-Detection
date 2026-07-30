#pragma once

#include "fraud_daemon/ring.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace fraud::daemon {

using ClockFn = std::uint64_t (*)() noexcept;
using BatchHandler = void (*)(void*, SlotMetadata* const*, std::size_t, std::uint64_t) noexcept;

struct BatcherConfig final {
  std::size_t max_batch_size{32};
  std::uint64_t sla_deadline_ns{2'000'000};
};

struct BatchMetrics final {
  std::uint64_t deadline_batches{};
  std::uint64_t full_batches{};
  std::uint64_t requests{};
};

// Fixed-capacity staging avoids heap allocation in poll_once(), the scheduler hot path.
template <std::size_t MaxBatch = 256>
class ContinuousBatcher final {
 public:
  ContinuousBatcher(SharedRing& ring, BatcherConfig config, ClockFn clock,
                    BatchHandler handler, void* handler_context) noexcept
      : ring_(ring), config_(config), clock_(clock), handler_(handler), context_(handler_context) {}

  bool poll_once() noexcept {
    if (config_.max_batch_size == 0 || config_.max_batch_size > MaxBatch || clock_ == nullptr || handler_ == nullptr) return false;
    while (count_ < config_.max_batch_size) {
      SlotMetadata* slot = ring_.try_consume();
      if (slot == nullptr) break;
      slots_[count_++] = slot;
      if (count_ == 1) oldest_enqueue_ns_ = slot->enqueue_ns;
    }
    if (count_ == 0) return false;
    const std::uint64_t now = clock_();
    const bool full = count_ == config_.max_batch_size;
    const bool overdue = now >= oldest_enqueue_ns_ && now - oldest_enqueue_ns_ >= config_.sla_deadline_ns;
    if (!full && !overdue) return false;
    handler_(context_, slots_.data(), count_, now);
    metrics_.requests += count_;
    if (full) ++metrics_.full_batches; else ++metrics_.deadline_batches;
    count_ = 0;
    oldest_enqueue_ns_ = 0;
    return true;
  }

  // Call at controlled shutdown to avoid stranding a partly formed batch.
  bool flush() noexcept {
    if (count_ == 0 || handler_ == nullptr || clock_ == nullptr) return false;
    handler_(context_, slots_.data(), count_, clock_());
    metrics_.requests += count_;
    ++metrics_.deadline_batches;
    count_ = 0;
    oldest_enqueue_ns_ = 0;
    return true;
  }
  [[nodiscard]] const BatchMetrics& metrics() const noexcept { return metrics_; }
  [[nodiscard]] std::size_t pending() const noexcept { return count_; }

 private:
  SharedRing& ring_;
  BatcherConfig config_;
  ClockFn clock_;
  BatchHandler handler_;
  void* context_;
  std::array<SlotMetadata*, MaxBatch> slots_{};
  std::size_t count_{};
  std::uint64_t oldest_enqueue_ns_{};
  BatchMetrics metrics_{};
};

std::uint64_t monotonic_time_ns() noexcept;

}  // namespace fraud::daemon
