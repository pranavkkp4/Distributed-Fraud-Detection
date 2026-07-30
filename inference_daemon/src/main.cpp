#include "fraud_daemon/batcher.hpp"
#include "fraud_daemon/cuda_graph.hpp"
#include "fraud_daemon/shared_memory.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
struct WorkerContext { fraud::daemon::SharedRing* ring; bool cuda_graphs_ready{}; };
void process_batch(void* opaque, fraud::daemon::SlotMetadata* const* slots, std::size_t count,
                   std::uint64_t now_ns) noexcept {
  auto& context = *static_cast<WorkerContext*>(opaque);
  // CPU reference decision. The native model adapter can replace this handler
  // without changing queue ownership or its no-allocation scheduler contract.
  std::uint32_t response_status = 200;
  if (context.cuda_graphs_ready) {
    const auto result = fraud::daemon::replay_inference_graph(count);
    // Keep the completed response deterministic and fail closed if an already
    // captured graph cannot replay. No allocator or lock is entered here.
    if (result.capability != fraud::daemon::CudaCapability::ready) response_status = 503;
  }
  for (std::size_t i = 0; i < count; ++i) {
    auto* slot = slots[i];
    const auto position = fraud::daemon::atomic_load(slot->sequence) - 1;
    context.ring->complete(*slot, position, response_status, 0, 0.0F, now_ns);
  }
}
bool parse_u32(const char* text, std::uint32_t& value) {
  char* end{};
  const auto parsed = std::strtoul(text, &end, 0);
  if (end == text || *end != '\0' || parsed > UINT32_MAX) return false;
  value = static_cast<std::uint32_t>(parsed);
  return true;
}
}

int main(int argc, char** argv) {
  std::uint32_t key = 0x46444950U, slots = 1024, slot_size = 4096;
  if (argc == 4 && (!parse_u32(argv[1], key) || !parse_u32(argv[2], slots) || !parse_u32(argv[3], slot_size))) {
    std::cerr << "invalid arguments\n"; return 2;
  }
  if (argc != 1 && argc != 4) {
    std::cerr << "usage: fraud_inference_daemon [sysv_key slot_count(power-of-two) slot_size]\n"; return 2;
  }
  std::string error;
  auto segment = fraud::daemon::SharedMemorySegment::create(key, slots, slot_size, error);
  if (!segment.valid()) { std::cerr << "daemon startup failed: " << error << '\n'; return 1; }
  fraud::daemon::SharedRing ring(*segment.header());
  bool cuda_graphs_ready = true;
  for (std::size_t batch_size = 1; batch_size <= 32; ++batch_size) {
    const auto graph = fraud::daemon::capture_inference_graph(batch_size);
    if (graph.capability != fraud::daemon::CudaCapability::ready) { cuda_graphs_ready = false; break; }
  }
  if (!cuda_graphs_ready) {
    fraud::daemon::shutdown_inference_graphs();
    std::cout << "CUDA Graph unavailable; using CPU reference handler\n";
  } else {
    std::cout << "CUDA Graph pre-captured batches 1..32\n";
  }
  WorkerContext context{&ring, cuda_graphs_ready};
  fraud::daemon::ContinuousBatcher<> batcher(ring, {32, 2'000'000}, fraud::daemon::monotonic_time_ns,
                                               process_batch, &context);
  std::cout << "ready: System V key=" << key << " slots=" << slots << " slot_size=" << slot_size << '\n';
  std::uint32_t idle_cycles{};
  while (fraud::daemon::atomic_load(segment.header()->shutdown) == 0) {
    if (batcher.poll_once()) { idle_cycles = 0; continue; }
    // Dedicated low-latency scheduler: bounded pause/yield rather than a fixed
    // sleep that can consume the 2ms SLA budget. Configurable by env if desired.
    if (++idle_cycles >= 64) { std::this_thread::yield(); idle_cycles = 0; }
  }
  (void)batcher.flush();
  fraud::daemon::shutdown_inference_graphs();
  segment.mark_for_removal();
  return 0;
}
