#pragma once

#include "fraud_daemon/shared_abi.hpp"

#include <cstdint>

namespace fraud::daemon {

// Bounded MPMC ring based on per-slot sequence numbers. It owns no memory and
// can therefore operate directly on a System V shared-memory segment.
class SharedRing final {
 public:
  explicit SharedRing(SharedHeader& header) : header_(header) {}

  bool try_produce(const void* payload, std::uint32_t size, std::uint64_t request_id,
                   std::uint64_t enqueue_ns, std::uint64_t* position_out = nullptr) noexcept;
  SlotMetadata* try_consume() noexcept;
  void complete(SlotMetadata& slot, std::uint64_t position, std::uint32_t status,
                std::uint32_t decision, float score, std::uint64_t complete_ns) noexcept;
  // The producer calls this only after it has acquired p+2 and copied its
  // response. This prevents a subsequent request from overwriting the result.
  bool release(SlotMetadata& slot, std::uint64_t position) noexcept;
  [[nodiscard]] const SharedHeader& header() const noexcept { return header_; }

 private:
  SharedHeader& header_;
};

inline bool SharedRing::try_produce(const void* payload, std::uint32_t size,
                                    std::uint64_t request_id, std::uint64_t enqueue_ns,
                                    std::uint64_t* position_out) noexcept {
  if (!valid_header(header_) || payload == nullptr) return false;
  // Reject before reserving an enqueue position: an invalid producer must not
  // create a permanent hole that would block consumers behind it.
  if (size > header_.slot_size - kSlotMetadataBytes) return false;
  std::uint64_t position = atomic_load(header_.enqueue);
  for (;;) {
    SlotMetadata* slot = slot_at(&header_, position);
    const std::uint64_t sequence = atomic_load(slot->sequence);
    const auto difference = static_cast<std::int64_t>(sequence) - static_cast<std::int64_t>(position);
    if (difference == 0) {
      std::uint64_t expected = position;
      if (atomic_compare_exchange(header_.enqueue, expected, position + 1)) {
        slot->payload_offset = kSlotMetadataBytes;
        slot->payload_size = size;
        slot->response_status = 0;
        slot->decision = 0;
        slot->score = 0.0F;
        slot->request_id = request_id;
        slot->enqueue_ns = enqueue_ns;
        slot->complete_ns = 0;
        std::memcpy(reinterpret_cast<std::byte*>(slot) + slot->payload_offset, payload, size);
        atomic_store(slot->sequence, position + 1);  // producer publishes p + 1
        atomic_fetch_add(header_.ready_count, 1);
        if (position_out != nullptr) *position_out = position;
        return true;
      }
      position = expected;
    } else if (difference < 0) {
      return false;
    } else {
      position = atomic_load(header_.enqueue);
    }
  }
}

inline SlotMetadata* SharedRing::try_consume() noexcept {
  if (!valid_header(header_)) return nullptr;
  std::uint64_t position = atomic_load(header_.dequeue);
  for (;;) {
    SlotMetadata* slot = slot_at(&header_, position);
    const std::uint64_t sequence = atomic_load(slot->sequence);
    const auto difference = static_cast<std::int64_t>(sequence) - static_cast<std::int64_t>(position + 1);
    if (difference == 0) {
      std::uint64_t expected = position;
      if (atomic_compare_exchange(header_.dequeue, expected, position + 1)) {
        atomic_fetch_add(header_.ready_count, static_cast<std::uint64_t>(-1));
        return slot;
      }
      position = expected;
    } else if (difference < 0) {
      return nullptr;
    } else {
      position = atomic_load(header_.dequeue);
    }
  }
}

inline void SharedRing::complete(SlotMetadata& slot, std::uint64_t position, std::uint32_t status,
                                 std::uint32_t decision, float score,
                                 std::uint64_t complete_ns) noexcept {
  slot.response_status = status;
  slot.decision = decision;
  slot.score = score;
  slot.complete_ns = complete_ns;
  // The response is durable at p+2. The producer must explicitly release it.
  atomic_store(slot.sequence, position + 2);
}

inline bool SharedRing::release(SlotMetadata& slot, std::uint64_t position) noexcept {
  if (!valid_header(header_) || atomic_load(slot.sequence) != position + 2) return false;
  atomic_store(slot.sequence, position + header_.slot_count);
  return true;
}

}  // namespace fraud::daemon
