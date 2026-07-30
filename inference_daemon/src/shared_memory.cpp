#include "fraud_daemon/shared_memory.hpp"

#include <cstring>
#include <limits>
#include <utility>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

namespace fraud::daemon {
namespace {
bool checked_size(std::uint32_t slots, std::uint32_t stride, std::size_t& result) noexcept {
  if (slots < 4 || (slots & (slots - 1)) != 0 || stride < kSlotMetadataBytes ||
      (stride % alignof(SlotMetadata)) != 0) return false;
  if (slots > (std::numeric_limits<std::size_t>::max() - kSharedHeaderBytes) / stride) return false;
  result = kSharedHeaderBytes + static_cast<std::size_t>(slots) * stride;
  return true;
}
}

bool initialize_segment(SharedHeader* header, std::size_t bytes, std::uint32_t slot_count,
                        std::uint32_t slot_size) noexcept {
  std::size_t required{};
  if (header == nullptr || !checked_size(slot_count, slot_size, required) || bytes < required) return false;
  std::memset(header, 0, required);
  header->magic = kSharedMagic;
  header->version = kSharedVersion;
  header->header_bytes = kSharedHeaderBytes;
  header->slot_count = slot_count;
  header->slot_size = slot_size;
  atomic_store(header->enqueue, 0, __ATOMIC_RELAXED);
  atomic_store(header->dequeue, 0, __ATOMIC_RELAXED);
  atomic_store(header->ready_count, 0, __ATOMIC_RELAXED);
  atomic_store(header->shutdown, 0);
  for (std::uint32_t i = 0; i < slot_count; ++i) {
    SlotMetadata* slot = slot_at(header, i);
    atomic_store(slot->sequence, i, __ATOMIC_RELAXED);  // initial seq=i
    slot->payload_offset = kSlotMetadataBytes;
  }
  return true;
}

SharedMemorySegment::~SharedMemorySegment() {
#if !defined(_WIN32)
  if (header_ != nullptr) (void)shmdt(header_);
#endif
}
SharedMemorySegment::SharedMemorySegment(SharedMemorySegment&& other) noexcept
    : id_(std::exchange(other.id_, -1)), header_(std::exchange(other.header_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}
SharedMemorySegment& SharedMemorySegment::operator=(SharedMemorySegment&& other) noexcept {
  if (this == &other) return *this;
#if !defined(_WIN32)
  if (header_ != nullptr) (void)shmdt(header_);
#endif
  id_ = std::exchange(other.id_, -1);
  header_ = std::exchange(other.header_, nullptr);
  bytes_ = std::exchange(other.bytes_, 0);
  return *this;
}

SharedMemorySegment SharedMemorySegment::create(std::uint32_t key, std::uint32_t slot_count,
                                                 std::uint32_t slot_size, std::string& error) {
  SharedMemorySegment segment;
  if (key == 0) { error = "shared memory key must be non-zero"; return segment; }
  if (!checked_size(slot_count, slot_size, segment.bytes_)) { error = "invalid shared-memory dimensions"; return segment; }
#if defined(_WIN32)
  (void)slot_count; (void)slot_size;
  error = "System V shared memory is not available on Windows; use the CPU unit tests or a Linux daemon";
#else
  segment.id_ = shmget(static_cast<key_t>(key), segment.bytes_, IPC_CREAT | IPC_EXCL | 0600);
  if (segment.id_ < 0) { error = "shmget failed (key may already exist)"; return segment; }
  void* attached = shmat(segment.id_, nullptr, 0);
  if (attached == reinterpret_cast<void*>(-1)) {
    error = "shmat failed";
    (void)shmctl(segment.id_, IPC_RMID, nullptr);
    segment.id_ = -1;
    return segment;
  }
  segment.header_ = static_cast<SharedHeader*>(attached);
  if (!initialize_segment(segment.header_, segment.bytes_, slot_count, slot_size)) {
    error = "failed to initialize shared-memory ABI";
    (void)shmdt(segment.header_);
    (void)shmctl(segment.id_, IPC_RMID, nullptr);
    segment.header_ = nullptr; segment.id_ = -1;
  }
#endif
  return segment;
}

void SharedMemorySegment::mark_for_removal() noexcept {
#if !defined(_WIN32)
  if (id_ >= 0) { (void)shmctl(id_, IPC_RMID, nullptr); id_ = -1; }
#endif
}

}  // namespace fraud::daemon
