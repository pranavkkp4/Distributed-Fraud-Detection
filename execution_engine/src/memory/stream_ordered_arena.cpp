#include "fraud_engine/execution_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace fraud_engine {
#ifdef FRAUD_ENGINE_WITH_CUDA
void* cuda_arena_create();
Status cuda_arena_allocate(void* context, std::size_t bytes, void** result);
Status cuda_arena_free(void* context, void* memory) noexcept;
Status cuda_arena_synchronize(void* context) noexcept;
void* cuda_arena_stream(void* context) noexcept;
int cuda_arena_device(void* context) noexcept;
void cuda_arena_destroy(void* context) noexcept;
#endif
struct StreamOrderedArena::Impl {
  struct Block { void* address; std::size_t alignment; std::size_t bytes; bool cuda; };
  std::vector<Block> blocks;
  std::size_t bytes{0};
  bool cuda_backed{false};
  void* cuda_context{nullptr};
  ~Impl() {
    (void)reset();
#ifdef FRAUD_ENGINE_WITH_CUDA
    cuda_arena_destroy(cuda_context);
#endif
  }
  Status reset() noexcept {
    Status first_error = Status::Ok();
    std::size_t retained_count = 0;
    std::size_t retained_bytes = 0;
    for (const Block& block : blocks) {
#ifdef FRAUD_ENGINE_WITH_CUDA
      if (block.cuda) {
        const Status free_status = cuda_arena_free(cuda_context, block.address);
        if (!free_status.ok()) {
          if (first_error.ok()) first_error = free_status;
          blocks[retained_count++] = block;
          retained_bytes += block.bytes;
        }
      } else
#endif
        ::operator delete(block.address, std::align_val_t(block.alignment));
    }
#ifdef FRAUD_ENGINE_WITH_CUDA
    Status synchronization_status = Status::Ok();
    if (cuda_backed) synchronization_status = cuda_arena_synchronize(cuda_context);
    if (first_error.ok() && !synchronization_status.ok()) first_error = synchronization_status;
#endif
    blocks.resize(retained_count);
    bytes = retained_bytes;
    return first_error;
  }
};

StreamOrderedArena::StreamOrderedArena(std::size_t initial_bytes) : impl_(std::make_unique<Impl>()) {
#ifdef FRAUD_ENGINE_WITH_CUDA
  impl_->cuda_context = cuda_arena_create();
  impl_->cuda_backed = impl_->cuda_context != nullptr;
#endif
  if (initial_bytes != 0) {
    void* unused = nullptr;
    const Status status = allocate(initial_bytes, alignof(std::max_align_t), &unused);
    if (!status.ok()) {
      if (status.code == StatusCode::kOutOfMemory) throw std::bad_alloc();
      throw std::runtime_error(status.message);
    }
  }
}
StreamOrderedArena::~StreamOrderedArena() = default;
StreamOrderedArena::StreamOrderedArena(StreamOrderedArena&&) noexcept = default;
StreamOrderedArena& StreamOrderedArena::operator=(StreamOrderedArena&&) noexcept = default;
Status StreamOrderedArena::allocate(std::size_t bytes, std::size_t alignment, void** result) {
  if (result == nullptr)
    return {StatusCode::kInvalidArgument, "arena result pointer is required"};
  *result = nullptr;
  if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return {StatusCode::kInvalidArgument, "bytes, result, and power-of-two alignment are required"};
  }
  try {
    const std::size_t actual_alignment = std::max(alignment, alignof(std::max_align_t));
    void* memory = nullptr;
    void* allocation_base = nullptr;
#ifdef FRAUD_ENGINE_WITH_CUDA
    const bool cuda = impl_->cuda_backed;
    if (cuda) {
      if (bytes > std::numeric_limits<std::size_t>::max() - (actual_alignment - 1))
        return {StatusCode::kInvalidArgument, "arena allocation size overflows"};
      const Status cuda_status = cuda_arena_allocate(
          impl_->cuda_context, bytes + actual_alignment - 1, &allocation_base);
      if (!cuda_status.ok()) {
        if (allocation_base != nullptr) {
          cuda_arena_free(impl_->cuda_context, allocation_base);
          (void)cuda_arena_synchronize(impl_->cuda_context);
        }
        return cuda_status;
      }
      const auto base = reinterpret_cast<std::uintptr_t>(allocation_base);
      memory = reinterpret_cast<void*>((base + actual_alignment - 1) & ~(actual_alignment - 1));
    } else
#else
    const bool cuda = false;
#endif
      allocation_base = memory = ::operator new(bytes, std::align_val_t(actual_alignment));
    try {
      impl_->blocks.push_back({allocation_base, actual_alignment, bytes, cuda});
    } catch (...) {
#ifdef FRAUD_ENGINE_WITH_CUDA
      if (cuda) {
        (void)cuda_arena_free(impl_->cuda_context, allocation_base);
        (void)cuda_arena_synchronize(impl_->cuda_context);
      } else
#endif
        ::operator delete(allocation_base, std::align_val_t(actual_alignment));
      throw;
    }
    impl_->bytes += bytes; *result = memory;
    return Status::Ok();
  } catch (const std::bad_alloc&) { *result = nullptr; return {StatusCode::kOutOfMemory, "arena allocation failed"}; }
}
Status StreamOrderedArena::synchronize() {
#ifdef FRAUD_ENGINE_WITH_CUDA
  if (impl_->cuda_backed) return cuda_arena_synchronize(impl_->cuda_context);
#endif
  return Status::Ok();
}
Status StreamOrderedArena::reset() noexcept { return impl_->reset(); }
std::size_t StreamOrderedArena::bytes_in_use() const noexcept { return impl_->bytes; }
bool StreamOrderedArena::is_cuda_backed() const noexcept { return impl_->cuda_backed; }
int StreamOrderedArena::device_ordinal() const noexcept {
#ifdef FRAUD_ENGINE_WITH_CUDA
  return cuda_arena_device(impl_->cuda_context);
#else
  return -1;
#endif
}
void* StreamOrderedArena::native_stream() const noexcept {
#ifdef FRAUD_ENGINE_WITH_CUDA
  return cuda_arena_stream(impl_->cuda_context);
#else
  return nullptr;
#endif
}
}  // namespace fraud_engine
