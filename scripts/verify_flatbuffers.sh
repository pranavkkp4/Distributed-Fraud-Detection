#!/usr/bin/env bash
set -Eeuo pipefail

flatc_bin="${1:-flatc}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generated="$(mktemp -d "${TMPDIR:-/tmp}/fraud-flatbuffers.XXXXXX")"
trap 'rm -rf "$generated"' EXIT

"$flatc_bin" --rust --cpp -o "$generated" "$repository_root/schemas/transaction.fbs"

# Official flatc archives can format generated source differently across host
# platforms. Compare compact token streams; the schema has no whitespace-bearing
# string defaults, and Rust's redundant `extern crate alloc` is host-variant.
python3 - "$generated" "$repository_root/api_gateway/src/generated" <<'PY'
from pathlib import Path
import sys

generated, checked = map(Path, sys.argv[1:])

def compact(path: Path, rust: bool = False) -> bytes:
    value = b"".join(path.read_bytes().split())
    return value.replace(b"externcratealloc;", b"") if rust else value

for name, rust in (("transaction_generated.rs", True), ("transaction_generated.h", False)):
    if compact(generated / name, rust) != compact(checked / name, rust):
        raise SystemExit(f"stale FlatBuffers binding: {name}")
PY
echo 'FlatBuffers Rust and C++ bindings match schemas/transaction.fbs.'
