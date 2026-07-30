#pragma once

#include "fraud_daemon/shared_abi.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace fraud::daemon {

class SharedMemorySegment final {
 public:
  SharedMemorySegment() = default;
  ~SharedMemorySegment();
  SharedMemorySegment(const SharedMemorySegment&) = delete;
  SharedMemorySegment& operator=(const SharedMemorySegment&) = delete;
  SharedMemorySegment(SharedMemorySegment&& other) noexcept;
  SharedMemorySegment& operator=(SharedMemorySegment&& other) noexcept;

  static SharedMemorySegment create(std::uint32_t key, std::uint32_t slot_count,
                                    std::uint32_t slot_size, std::string& error);
  [[nodiscard]] SharedHeader* header() const noexcept { return header_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool valid() const noexcept { return header_ != nullptr; }
  void mark_for_removal() noexcept;

 private:
  int id_{-1};
  SharedHeader* header_{};
  std::size_t bytes_{};
};

bool initialize_segment(SharedHeader* header, std::size_t bytes, std::uint32_t slot_count,
                        std::uint32_t slot_size) noexcept;

}  // namespace fraud::daemon
