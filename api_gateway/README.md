# Shared-memory Rust API gateway

This Linux-only service attaches to the System V shared-memory segment owned by
`inference_daemon`; it does **not** create or remove that segment.  It accepts a
bounded JSON transaction over HTTP, authenticates a bearer token in constant
time, serializes FlatBuffers directly into the selected shared-memory slot, and
waits only until its configured deadline.  Timed-out work is reclaimed by a
bounded background reaper after the daemon signals completion.

## Build and run

```bash
cd api_gateway
cargo test
FRAUD_SHM_KEY=1234 FRAUD_GATEWAY_TOKEN='replace-me-securely' cargo run --release
```

Required environment variables are `FRAUD_SHM_KEY` and
`FRAUD_GATEWAY_TOKEN` (16--256 bytes). `FRAUD_GATEWAY_ADDR` defaults to `127.0.0.1:8081` and
`FRAUD_GATEWAY_DEADLINE_MS` defaults to 10 ms (capped at 50 ms).  Terminate TLS
at a local authenticated reverse proxy; do not bind this intentionally tiny
gateway directly to an untrusted network.

```bash
curl -sS http://127.0.0.1:8081/v1/transactions \
  -H 'Authorization: Bearer replace-me-securely' -H 'content-type: application/json' \
  --data '{"transaction_id":"tx-1","account_id":"acct-1","amount_micros":12340000,"currency":"USD","occurred_at_ns":1735689600000000000}'
```

`amount_micros` is a required signed integer, not a floating-point amount. The
gateway accepts values in `[-9000000000000000000, 9000000000000000000]` to
retain exact monetary representation and leave headroom below `i64::MAX`.

## ABI

The header uses magic `0x46444950`, version `1`, and is exactly 320 bytes.
Metadata words are at offsets 0/4/8/12/16; atomics are `enqueue@64`,
`dequeue@128`, `ready_count@192`, and `shutdown@256`.  The ring capacity must
be a power of two and at least four. Slots have `sequence`
at 0; payload offset/size at 8/12; response status/decision/score at 16/20/24;
and request/enqueue/complete timestamps at 32/40/48.  The FlatBuffer offset is
slot-relative and always at least 64. `enqueue_ns` uses Linux
`CLOCK_MONOTONIC`, matching the daemon's `steady_clock` contract. Both
processes must use naturally aligned, process-shared lock-free atomics on a
64-bit little-endian Linux x86_64 or aarch64 target.

The producer sequence protocol is: claim `seq == p`, write a slot, then store
`p + 1` with release ordering; the daemon completes with `p + 2`; the gateway
reads after an acquire load and stores `p + capacity` to release the slot.
If serialization unwinds or the absolute request deadline expires before
publication, the gateway publishes a cancellation record (`payload_size = 0`,
`response_status = 0xfffffffe`) and gives it to the reclaimer. The daemon must
consume this record without inference, decrement `ready_count`, and complete it
with `p + 2`; otherwise no process can safely skip the claimed ring position.

Daemon response statuses are closed HTTP-like values: `200=scored`,
`422=invalid_input`, and `503=unavailable`. The gateway preserves that numeric
status in its JSON response and uses it as the HTTP response status. Unknown
and incomplete values are treated as a gateway error, while the slot is still
released.

## Schema generation

`../schemas/transaction.fbs` is the single wire contract.  When `flatc` is on
`PATH`, `build.rs` generates Rust and C++ bindings into Cargo's build output
and verifies that the checked-in Rust binding at
`src/generated/transaction_generated.rs` is current. That generated binding is
used directly by the request writer; its C++ companion is checked in alongside
it for the daemon build.
For release CI require it explicitly:

```bash
FRAUD_REQUIRE_FLATC=1 cargo build --release
flatc --rust --cpp -o generated ../schemas/transaction.fbs
```

The generated table wrapper is constructed with
`FlatBufferBuilder::new_in(SharedSlotAllocator)`.
That allocator is fixed-size: it cannot grow and never falls back to a serialized
`Vec<u8>`. The current FlatBuffers Rust API reports an allocator-growth failure
by panicking rather than returning a `try_*` error, so the gateway uses a
capacity preflight, `catch_unwind`, and an RAII cancellation guard to prevent a
lost ring position. HTTP receive buffers still necessarily exist; JSON string
fields are borrowed from them while the FlatBuffer is constructed in shared
memory.
