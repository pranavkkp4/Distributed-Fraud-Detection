#include "fraud_daemon/batcher.hpp"
#include "fraud_daemon/memory_pool.hpp"
#include "fraud_daemon/shared_memory.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <thread>
#include <vector>

namespace {
std::atomic<std::uint64_t> allocations{0};
void* operator_new_impl(std::size_t bytes) { allocations.fetch_add(1, std::memory_order_relaxed); if (void* ptr = std::malloc(bytes)) return ptr; throw std::bad_alloc{}; }
}
void* operator new(std::size_t bytes) { return operator_new_impl(bytes); }
void* operator new[](std::size_t bytes) { return operator_new_impl(bytes); }
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

namespace {
using namespace fraud::daemon;
int failures{};
#define CHECK(expr) do { if (!(expr)) { std::cerr << "FAIL " #expr " at " << __LINE__ << '\n'; ++failures; } } while (false)

struct Segment {
  std::vector<std::uint64_t> words;
  SharedHeader* header;
  Segment(std::uint32_t count, std::uint32_t stride)
      : words((kSharedHeaderBytes + count * stride + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)),
        header(reinterpret_cast<SharedHeader*>(words.data())) {
    CHECK(initialize_segment(header, words.size() * sizeof(std::uint64_t), count, stride));
  }
};
std::uint64_t fake_now{};
std::uint64_t fake_clock() noexcept { return fake_now; }
struct HandlerState { SharedRing* ring{}; std::size_t last_count{}; };
void handler(void* opaque, SlotMetadata* const* slots, std::size_t count, std::uint64_t now) noexcept {
  auto& state = *static_cast<HandlerState*>(opaque); state.last_count = count;
  for (std::size_t i = 0; i < count; ++i) { const auto position = atomic_load(slots[i]->sequence) - 1; state.ring->complete(*slots[i], position, 200, 1, .8F, now); }
}

void test_abi_and_ring() {
  CHECK(sizeof(SharedHeader) == 320); CHECK(offsetof(SharedHeader, enqueue) == 64); CHECK(offsetof(SharedHeader, shutdown) == 256);
  Segment segment(4, 128); SharedRing ring(*segment.header); const std::array<std::byte, 2> payload{std::byte{0x2}, std::byte{0x3}};
  std::uint64_t position{}; CHECK(ring.try_produce(payload.data(), payload.size(), 7, 10, &position)); CHECK(position == 0);
  SlotMetadata* received = ring.try_consume(); CHECK(received != nullptr); CHECK(received->request_id == 7); CHECK(received->payload_offset >= 64);
  CHECK(std::memcmp(reinterpret_cast<std::byte*>(received) + received->payload_offset, payload.data(), payload.size()) == 0);
  ring.complete(*received, position, 200, 1, .5F, 20); CHECK(atomic_load(received->sequence) == 2); // consumer completes p+2
  // Fill positions 1,2,3; position 4 must not overwrite response at position 0.
  CHECK(ring.try_produce(payload.data(), payload.size(), 8, 21)); CHECK(ring.try_produce(payload.data(), payload.size(), 9, 22));
  CHECK(ring.try_produce(payload.data(), payload.size(), 10, 23)); CHECK(!ring.try_produce(payload.data(), payload.size(), 11, 24));
  CHECK(received->request_id == 7 && received->response_status == 200); CHECK(ring.release(*received, position));
  CHECK(ring.try_produce(payload.data(), payload.size(), 11, 24));
}

void test_mpmc_bounded() {
  Segment segment(64, 128); SharedRing ring(*segment.header); constexpr int total = 5000; std::atomic<int> produced{}; std::atomic<int> consumed{};
  auto producer = [&] { std::uint32_t payload = 42; for (;;) { const int i = produced.fetch_add(1); if (i >= total) break; while (!ring.try_produce(&payload, sizeof(payload), static_cast<std::uint64_t>(i), i)) std::this_thread::yield(); } };
  auto consumer = [&] { while (consumed.load() < total) { if (auto* slot = ring.try_consume()) { const auto p = atomic_load(slot->sequence) - 1; ring.complete(*slot, p, 200, 0, 0, 1); CHECK(ring.release(*slot, p)); consumed.fetch_add(1); } else std::this_thread::yield(); } };
  std::thread p1(producer), p2(producer), c1(consumer), c2(consumer); p1.join(); p2.join(); c1.join(); c2.join(); CHECK(consumed == total);
}

void test_deadline_and_hot_path() {
  Segment segment(8, 128); SharedRing ring(*segment.header); HandlerState state{&ring};
  ContinuousBatcher<> batcher(ring, {4, 100}, fake_clock, handler, &state); std::uint32_t payload = 1;
  fake_now = 10; CHECK(ring.try_produce(&payload, sizeof(payload), 1, fake_now));
  const auto before = allocations.load(std::memory_order_relaxed);
  CHECK(!batcher.poll_once()); fake_now = 110; CHECK(batcher.poll_once());
  CHECK(state.last_count == 1); CHECK(allocations.load(std::memory_order_relaxed) == before);
  for (int i = 0; i < 4; ++i) CHECK(ring.try_produce(&payload, sizeof(payload), i + 2, fake_now));
  CHECK(batcher.poll_once()); CHECK(state.last_count == 4); CHECK(batcher.metrics().deadline_batches == 1 && batcher.metrics().full_batches == 1);
}

void test_memory_pool() {
  MemoryPool pool(4096); auto* value = pool.make<std::uint64_t>(55); CHECK(*value == 55); CHECK(pool.used() >= sizeof(*value));
  ArenaAllocator<int> allocator(pool); int* number = allocator.allocate(1); *number = 9; CHECK(*number == 9);
}
}
int main() { test_abi_and_ring(); test_mpmc_bounded(); test_deadline_and_hot_path(); test_memory_pool(); return failures == 0 ? 0 : 1; }
