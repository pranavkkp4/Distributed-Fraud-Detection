#!/usr/bin/env bash
set -Eeuo pipefail

flatc_bin="${1:-flatc}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generated="$(mktemp -d "${TMPDIR:-/tmp}/fraud-flatbuffers.XXXXXX")"
trap 'rm -rf "$generated"' EXIT

"$flatc_bin" --rust --cpp -o "$generated" "$repository_root/schemas/transaction.fbs"
cmp "$generated/transaction_generated.rs" \
  "$repository_root/api_gateway/src/generated/transaction_generated.rs"
cmp "$generated/transaction_generated.h" \
  "$repository_root/api_gateway/src/generated/transaction_generated.h"
echo 'FlatBuffers Rust and C++ bindings match schemas/transaction.fbs.'
