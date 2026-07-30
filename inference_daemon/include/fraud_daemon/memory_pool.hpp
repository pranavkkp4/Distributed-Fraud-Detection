#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace fraud::daemon {

class MemoryPool final {
 public:
  explicit MemoryPool(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));
  ~MemoryPool();
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment);
  [[nodiscard]] std::size_t capacity() const noexcept { return bytes_; }
  [[nodiscard]] std::size_t used() const noexcept { return cursor_.load(std::memory_order_relaxed); }

  template <class T, class... Args>
  [[nodiscard]] T* make(Args&&... args) {
    return ::new (allocate(sizeof(T), alignof(T))) T(std::forward<Args>(args)...);
  }

 private:
  void* base_{};
  std::size_t bytes_{};
  std::size_t allocation_bytes_{};
  std::atomic<std::size_t> cursor_{0};
  bool mapped_{};
};

template <class T>
class ArenaAllocator {
 public:
  using value_type = T;
  explicit ArenaAllocator(MemoryPool& pool) noexcept : pool_(&pool) {}
  template <class U> ArenaAllocator(const ArenaAllocator<U>& other) noexcept : pool_(other.pool()) {}
  T* allocate(std::size_t n) {
    if (n > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_array_new_length{};
    return static_cast<T*>(pool_->allocate(n * sizeof(T), alignof(T)));
  }
  void deallocate(T*, std::size_t) noexcept {}
  MemoryPool* pool() const noexcept { return pool_; }
  template <class U> bool operator==(const ArenaAllocator<U>& rhs) const noexcept { return pool_ == rhs.pool(); }
 private: MemoryPool* pool_;
};

}  // namespace fraud::daemon
