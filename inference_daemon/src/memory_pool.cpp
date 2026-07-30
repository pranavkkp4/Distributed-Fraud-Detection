#include "fraud_daemon/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fraud::daemon {
namespace {
std::size_t checked_round_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > (std::numeric_limits<std::size_t>::max)() - (alignment - 1)) {
    throw std::invalid_argument("invalid memory-pool alignment or size");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}
}  // namespace

MemoryPool::MemoryPool(std::size_t bytes, std::size_t alignment) : bytes_(bytes) {
  if (bytes == 0) throw std::invalid_argument("memory pool must be non-empty");
#if defined(_WIN32)
  (void)alignment;
  base_ = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  allocation_bytes_ = bytes;
  mapped_ = true;
#else
  const long page_size = sysconf(_SC_PAGESIZE);
  const auto page = page_size > 0 ? static_cast<std::size_t>(page_size) : alignment;
  allocation_bytes_ = checked_round_up(bytes, (std::max)(alignment, page));
  base_ = mmap(nullptr, allocation_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base_ == MAP_FAILED) base_ = nullptr;
  mapped_ = true;
#endif
  if (base_ == nullptr) throw std::bad_alloc{};
}

MemoryPool::~MemoryPool() {
  if (base_ == nullptr) return;
#if defined(_WIN32)
  (void)VirtualFree(base_, 0, MEM_RELEASE);
#else
  (void)munmap(base_, allocation_bytes_);
#endif
}

void* MemoryPool::allocate(std::size_t bytes, std::size_t alignment) {
  const auto current = cursor_.load(std::memory_order_relaxed);
  for (std::size_t position = current;;) {
    const auto aligned = checked_round_up(position, alignment);
    if (aligned > bytes_ || bytes > bytes_ - aligned) throw std::bad_alloc{};
    const auto next = aligned + bytes;
    if (cursor_.compare_exchange_weak(position, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return static_cast<std::byte*>(base_) + aligned;
    }
  }
}

}  // namespace fraud::daemon
