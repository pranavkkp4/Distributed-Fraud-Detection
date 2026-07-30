#include "fraud_daemon/transaction.hpp"

#include "transaction_generated.h"

#include <cstddef>
#include <cstdint>

namespace fraud::daemon {
namespace {

std::string_view view_of(const flatbuffers::String* value) noexcept {
  return value == nullptr ? std::string_view{} : std::string_view(value->c_str(), value->size());
}

}  // namespace

bool decode_transaction(const SharedHeader& header, const SlotMetadata& slot,
                        TransactionView& result) noexcept {
  if (!valid_header(header) || slot.payload_size == 0 ||
      slot.payload_offset < kSlotMetadataBytes || slot.payload_offset > header.slot_size ||
      slot.payload_size > header.slot_size - slot.payload_offset) {
    return false;
  }

  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&slot) + slot.payload_offset;
  flatbuffers::Verifier verifier(bytes, slot.payload_size);
  if (!fraud::ipc::VerifyTransactionBuffer(verifier)) return false;

  const fraud::ipc::Transaction* transaction = fraud::ipc::GetTransaction(bytes);
  if (transaction == nullptr || transaction->transaction_id() == nullptr ||
      transaction->account_id() == nullptr || transaction->currency() == nullptr) {
    return false;
  }

  result.request_id = transaction->request_id();
  result.transaction_id = view_of(transaction->transaction_id());
  result.account_id = view_of(transaction->account_id());
  result.amount_micros = transaction->amount_micros();
  result.currency = view_of(transaction->currency());
  result.occurred_at_ns = transaction->occurred_at_ns();
  result.merchant_category = view_of(transaction->merchant_category());
  return result.request_id == slot.request_id;
}

std::uint32_t reference_decision(const TransactionView& transaction) noexcept {
  // $10.00 in micro-units. The comparison has no overflow path.
  return transaction.amount_micros >= 10'000'000 ? 1U : 0U;
}

float reference_score(const TransactionView& transaction) noexcept {
  return reference_decision(transaction) == 0U ? 0.1F : 0.9F;
}

}  // namespace fraud::daemon
