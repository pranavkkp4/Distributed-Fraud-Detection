#pragma once

#include "fraud_daemon/shared_abi.hpp"

#include <cstdint>
#include <string_view>

namespace fraud::daemon {

// Views point into the slot payload and remain valid only until that slot is
// released back to the producer ring. They deliberately avoid allocations.
struct TransactionView final {
  std::uint64_t request_id{};
  std::string_view transaction_id{};
  std::string_view account_id{};
  std::int64_t amount_micros{};
  std::string_view currency{};
  std::uint64_t occurred_at_ns{};
  std::string_view merchant_category{};
};

// Validates slot bounds, the FRTX file identifier, and all FlatBuffers offsets
// before returning borrowed fields. A zero-size payload is a cancellation
// record and is rejected here for the caller to handle separately.
[[nodiscard]] bool decode_transaction(const SharedHeader& header, const SlotMetadata& slot,
                                      TransactionView& result) noexcept;

// A deliberately simple deterministic CPU reference result. It is used for
// the integration contract until the model adapter owns the decoded features.
[[nodiscard]] std::uint32_t reference_decision(const TransactionView& transaction) noexcept;
[[nodiscard]] float reference_score(const TransactionView& transaction) noexcept;

}  // namespace fraud::daemon
