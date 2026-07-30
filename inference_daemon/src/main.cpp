#include "fraud_daemon/batcher.hpp"
#include "fraud_daemon/cuda_graph.hpp"
#include "fraud_daemon/shared_memory.hpp"
#include "fraud_daemon/transaction.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
volatile std::sig_atomic_t stop_requested{};
void handle_signal(int) noexcept { stop_requested = 1; }

struct WorkerContext { fraud::daemon::SharedRing* ring; bool cuda_graphs_ready{}; };
void process_batch(void* opaque, fraud::daemon::SlotMetadata* const* slots, std::size_t count,
                   std::uint64_t now_ns) noexcept {
  auto& context = *static_cast<WorkerContext*>(opaque);
  std::size_t inference_count{};
  std::array<fraud::daemon::TransactionView, 256> transactions{};
  std::array<bool, 256> valid_transactions{};
  for (std::size_t i = 0; i < count; ++i) {
    // A producer that fails after reserving a position publishes a zero-length
    // cancellation record so the queue cannot develop a permanent hole.
    valid_transactions[i] = slots[i]->payload_size != 0 &&
                            fraud::daemon::decode_transaction(context.ring->header(), *slots[i], transactions[i]);
    if (valid_transactions[i]) ++inference_count;
  }
  // CPU reference decision. The native model adapter can replace this handler
  // without changing queue ownership or its no-allocation scheduler contract.
  std::uint32_t response_status = 200;
  fraud::daemon::CudaGraphStatus launched{fraud::daemon::CudaCapability::unavailable,
                                          "no active CUDA work"};
  if (context.cuda_graphs_ready && inference_count != 0) {
    launched = fraud::daemon::launch_inference_graph(inference_count);
    if (launched.capability != fraud::daemon::CudaCapability::ready) response_status = 503;
  }
  // Prepare response metadata while the nonblocking graph runs. This stack-only
  // work is deliberately placed between launch and synchronization so a Systems
  // trace can verify genuine host/device overlap in the daemon hot path.
  std::array<std::uint64_t, 256> positions{};
  std::array<std::uint32_t, 256> statuses{};
  std::array<std::uint32_t, 256> decisions{};
  std::array<float, 256> scores{};
  for (std::size_t i = 0; i < count; ++i) {
    positions[i] = fraud::daemon::atomic_load(slots[i]->sequence) - 1;
    statuses[i] = valid_transactions[i] ? response_status : 422U;
    if (valid_transactions[i] && response_status == 200U) {
      decisions[i] = fraud::daemon::reference_decision(transactions[i]);
      scores[i] = fraud::daemon::reference_score(transactions[i]);
    }
  }
  if (launched.capability == fraud::daemon::CudaCapability::ready &&
      fraud::daemon::synchronize_inference_graph(inference_count).capability !=
          fraud::daemon::CudaCapability::ready) {
    for (std::size_t i = 0; i < count; ++i) {
      if (valid_transactions[i]) {
        statuses[i] = 503;
        decisions[i] = 0;
        scores[i] = 0.0F;
      }
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    context.ring->complete(*slots[i], positions[i], statuses[i], decisions[i], scores[i], now_ns);
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
  (void)std::signal(SIGINT, handle_signal);
  (void)std::signal(SIGTERM, handle_signal);
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
  while (stop_requested == 0 && fraud::daemon::atomic_load(segment.header()->shutdown) == 0) {
    if (batcher.poll_once()) { idle_cycles = 0; continue; }
    // Dedicated low-latency scheduler: bounded pause/yield rather than a fixed
    // sleep that can consume the 2ms SLA budget. Configurable by env if desired.
    if (++idle_cycles >= 64) { std::this_thread::yield(); idle_cycles = 0; }
  }
  fraud::daemon::atomic_store(segment.header()->shutdown, 1);
  (void)batcher.flush();
  fraud::daemon::shutdown_inference_graphs();
  segment.mark_for_removal();
  return 0;
}
