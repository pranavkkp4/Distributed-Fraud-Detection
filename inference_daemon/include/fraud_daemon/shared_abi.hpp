#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace fraud::daemon {

#if defined(_MSC_VER)
inline constexpr int __ATOMIC_RELAXED = 0;
inline constexpr int __ATOMIC_ACQUIRE = 2;
inline constexpr int __ATOMIC_RELEASE = 3;
#endif

inline constexpr std::uint32_t kSharedMagic = 0x46444950U;  // "FDIP"
inline constexpr std::uint32_t kSharedVersion = 1;
inline constexpr std::uint32_t kSharedHeaderBytes = 320;
inline constexpr std::size_t kSlotMetadataBytes = 64;

// These are plain, aligned integers deliberately.  std::atomic<T>'s object
// representation is not a stable cross-language or cross-process ABI.
struct alignas(8) AtomicU64 { std::uint64_t value; };
struct alignas(4) AtomicU32 { std::uint32_t value; };

inline std::uint64_t atomic_load(const AtomicU64& source,
                                 int order = __ATOMIC_ACQUIRE) noexcept {
#if defined(_MSC_VER)
  (void)order;
  // InterlockedCompareExchange64 is an atomic acquire/release operation.
  return static_cast<std::uint64_t>(_InterlockedCompareExchange64(
      reinterpret_cast<volatile long long*>(const_cast<std::uint64_t*>(&source.value)), 0, 0));
#else
  return __atomic_load_n(&source.value, order);
#endif
}
inline void atomic_store(AtomicU64& destination, std::uint64_t value,
                         int order = __ATOMIC_RELEASE) noexcept {
#if defined(_MSC_VER)
  (void)order;
  _InterlockedExchange64(reinterpret_cast<volatile long long*>(&destination.value),
                         static_cast<long long>(value));
#else
  __atomic_store_n(&destination.value, value, order);
#endif
}
inline bool atomic_compare_exchange(AtomicU64& destination, std::uint64_t& expected,
                                    std::uint64_t desired) noexcept {
#if defined(_MSC_VER)
  const auto observed = static_cast<std::uint64_t>(_InterlockedCompareExchange64(
      reinterpret_cast<volatile long long*>(&destination.value), static_cast<long long>(desired),
      static_cast<long long>(expected)));
  if (observed == expected) return true;
  expected = observed;
  return false;
#else
  return __atomic_compare_exchange_n(&destination.value, &expected, desired, false,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
}
inline std::uint32_t atomic_load(const AtomicU32& source) noexcept {
#if defined(_MSC_VER)
  return static_cast<std::uint32_t>(_InterlockedCompareExchange(
      reinterpret_cast<volatile long*>(const_cast<std::uint32_t*>(&source.value)), 0, 0));
#else
  return __atomic_load_n(&source.value, __ATOMIC_ACQUIRE);
#endif
}
inline void atomic_store(AtomicU32& destination, std::uint32_t value) noexcept {
#if defined(_MSC_VER)
  _InterlockedExchange(reinterpret_cast<volatile long*>(&destination.value), static_cast<long>(value));
#else
  __atomic_store_n(&destination.value, value, __ATOMIC_RELEASE);
#endif
}
inline std::uint64_t atomic_fetch_add(AtomicU64& destination, std::uint64_t value) noexcept {
#if defined(_MSC_VER)
  return static_cast<std::uint64_t>(_InterlockedExchangeAdd64(
      reinterpret_cast<volatile long long*>(&destination.value), static_cast<long long>(value)));
#else
  return __atomic_fetch_add(&destination.value, value, __ATOMIC_ACQ_REL);
#endif
}

struct alignas(8) SharedHeader final {
  std::uint32_t magic;          // 0
  std::uint32_t version;        // 4
  std::uint32_t header_bytes;   // 8
  std::uint32_t slot_count;     // 12
  std::uint32_t slot_size;      // 16
  std::array<std::byte, 44> reserved_before_enqueue;
  AtomicU64 enqueue;            // 64
  std::array<std::byte, 56> reserved_before_dequeue;
  AtomicU64 dequeue;            // 128
  std::array<std::byte, 56> reserved_before_ready;
  AtomicU64 ready_count;        // 192
  std::array<std::byte, 56> reserved_before_shutdown;
  AtomicU32 shutdown;           // 256
  std::array<std::byte, 60> reserved_end;
};

struct alignas(8) SlotMetadata final {
  AtomicU64 sequence;            // 0
  std::uint32_t payload_offset;  // 8, >= 64
  std::uint32_t payload_size;    // 12
  std::uint32_t response_status; // 16
  std::uint32_t decision;        // 20
  float score;                   // 24
  std::uint32_t reserved;        // 28
  std::uint64_t request_id;      // 32
  std::uint64_t enqueue_ns;      // 40
  std::uint64_t complete_ns;     // 48
  std::array<std::byte, 8> reserved_end;
};

static_assert(sizeof(SharedHeader) == kSharedHeaderBytes);
static_assert(offsetof(SharedHeader, magic) == 0 && offsetof(SharedHeader, version) == 4);
static_assert(offsetof(SharedHeader, header_bytes) == 8 && offsetof(SharedHeader, slot_count) == 12);
static_assert(offsetof(SharedHeader, slot_size) == 16 && offsetof(SharedHeader, enqueue) == 64);
static_assert(offsetof(SharedHeader, dequeue) == 128 && offsetof(SharedHeader, ready_count) == 192);
static_assert(offsetof(SharedHeader, shutdown) == 256);
static_assert(sizeof(SlotMetadata) == kSlotMetadataBytes);
static_assert(offsetof(SlotMetadata, sequence) == 0 && offsetof(SlotMetadata, payload_offset) == 8);
static_assert(offsetof(SlotMetadata, payload_size) == 12 && offsetof(SlotMetadata, response_status) == 16);
static_assert(offsetof(SlotMetadata, decision) == 20 && offsetof(SlotMetadata, score) == 24);
static_assert(offsetof(SlotMetadata, request_id) == 32 && offsetof(SlotMetadata, enqueue_ns) == 40);
static_assert(offsetof(SlotMetadata, complete_ns) == 48);

inline SlotMetadata* slot_at(SharedHeader* header, std::uint64_t position) noexcept {
  auto* base = reinterpret_cast<std::byte*>(header) + header->header_bytes;
  return reinterpret_cast<SlotMetadata*>(base + (position % header->slot_count) * header->slot_size);
}
inline const SlotMetadata* slot_at(const SharedHeader* header, std::uint64_t position) noexcept {
  return slot_at(const_cast<SharedHeader*>(header), position);
}

inline bool valid_header(const SharedHeader& header) noexcept {
  return header.magic == kSharedMagic && header.version == kSharedVersion &&
         header.header_bytes == kSharedHeaderBytes && header.slot_count >= 4 &&
         (header.slot_count & (header.slot_count - 1)) == 0 &&
         header.slot_size >= kSlotMetadataBytes && (header.slot_size % alignof(SlotMetadata)) == 0;
}

}  // namespace fraud::daemon
