# Implementation status

The repository separates executable behavior from measured optimization claims.
"Implemented" below means source plus automated correctness coverage; it does
not mean a latency target has been demonstrated on arbitrary hardware. The
[optimization evidence log](../optimization_history.md) records the narrower
contexts in which latency was actually observed.

| Area | Implemented baseline | Deliberate boundary |
| --- | --- | --- |
| Model lifecycle | Fixed-shape 16x32 transformer, deterministic samples, activation calibration, immutable artifact hashes, ONNX export, optional TensorRT build | Training data and model quality are synthetic; a production registry/signing adapter is not included |
| CPU execution | FP32 projection and short-context attention microkernel reference with finite-input validation | It is a component-level correctness oracle, not the full exported model or a latency target |
| Native IPC daemon | Linux System V segment ownership, fixed cross-language ABI, startup arena component, bounded sequence ring, explicit producer response release, cancellation recovery, in-place generated-binding FRTX verification/decoding with slot bounds checks, and a dedicated deadline/full-batch scheduler with allocation-counter tests | The main loop consumes shared slots directly rather than allocating request objects from the arena; its response is a deterministic reference decision, it does not load ONNX/TensorRT, and the allocation test covers the scheduler hot path rather than every process path |
| Rust transaction gateway | Bearer-authenticated HTTP, bounded JSON validation with integer micros, FlatBuffers construction in fixed shared-slot storage, process-shared signaling, deadlines, and RAII/reaper slot recovery | JSON/HTTP handling necessarily allocates outside the shared slot; System V IPC is Linux-only, TLS must terminate at a trusted local proxy, and this is not a claim of network-level zero-copy |
| CUDA daemon components | Startup-persistent pinned/device buffers and streams, pre-captured CUDA Graphs for supported batch sizes, matching replay, capability reporting, and a concrete SM80 CUTLASS INT8-to-INT32 GEMM | Captures use a one-element FP32 forward stub and are not connected to transaction payloads or the exported transformer; local measurements are component wall times, not API/model latency |
| Go load tester | Seeded Poisson arrivals, bounded concurrency, transaction/entity payload modes, HDR histogram percentiles, JSON/CSV provenance, and a measured SVG CDF | It is a measurement instrument, not proof that an offered rate was sustained on other machines; non-2xx and transport failures remain observations rather than being discarded |
| Go serving plane | Authenticated HTTPS gateway, authenticated TLS 1.3 Redis, phase deadlines, bounded continuous batching, mTLS gRPC workers, replica health/circuit routing, deterministic fail-closed response, health, and metrics | This is a separate network-serving path from the Rust/System V gateway; the local worker is a deterministic scorer and the C++/TensorRT worker adapter remains a milestone |
| Streaming | Authenticated TLS 1.3 Kafka/Redis transports, partition-order consumption with bounded fetch/process/commit retries, idempotent event-time windows, capacity-bounded state, fixed-width materialization, and local mode | Capacity exhaustion halts safely; poison records halt by default or use an explicitly configured synchronous DLQ, while exactly-once deployment still requires Kafka transactions and durable state |
| Deployment and validation | Compose topology, Minikube manifests, Prometheus/Grafana assets, CPU/CUDA unit tests, generated-schema checks, live authenticated IPC CI, native/API regression suites, and reproducible profiling workflows | Hosted CI validates behavior, not production capacity; local image tags must be replaced with registry digests during controlled promotion |

The native path is described in the
[inference daemon guide](../../inference_daemon/README.md),
[Rust gateway guide](../../api_gateway/README.md), and
[load tester guide](../../load_tester/README.md).

## Shared-memory ownership and status semantics

The System V ring keeps a completed response at sequence `p + 2` until the
originating producer reads it and releases the slot at `p + capacity`; a later
request cannot overwrite an unread result. Producer serialization cancellation
is published as an empty record so the daemon can consume the queue position and
complete it with status 422 rather than leaving a permanent hole. The current
daemon response contract uses 200 for a completed reference response, 422 for a
cancelled/invalid record, and 503 when the configured CUDA replay cannot run.

The scheduler stages requests in fixed-capacity storage, triggers on either the
oldest SLA deadline or maximum batch size, and submits the resulting variable
batch. This establishes bounded mechanics; it does not prove that all libc,
HTTP, logging, driver, or model-adapter paths are allocation-free.

## Numeric backend terminology

There are two native CUDA surfaces and their results must not be conflated:

- `execution_engine` provides capability-selected FP16/E4M3 reference
  microkernels with FP32 accumulation and explicit CPU fallback reporting.
- `inference_daemon` provides captured-graph lifecycle and a CUTLASS INT8 GEMM
  component, but its captured forward operation is still a reference stub.

`preferred_backend` describes what device capability permits. `backend` on an
`execution_engine` result describes what actually produced that result. A failed
CUDA path is recomputed by its FP32 reference and reported as CPU with
`fallback_used`; it is never labeled FP16 or FP8 merely because a compatible
device was detected.

The E4M3 execution-engine path stores operands in FP8 and accumulates scalar
products in FP32; it is not a Tensor Core GEMM. The daemon CUTLASS component is
a real SM80 TensorOp INT8-to-INT32 GEMM, but it is only a component oracle until
the model adapter supplies calibrated transformer operands. Neither component
currently establishes full-model Tensor Core throughput.

## Performance status

One CPU-only GitHub-hosted run measured the full local HTTP/shared-memory path
using the deterministic daemon stub: 15,000/15,000 successful responses at an
observed 1,505.70 responses/s, with p50 2,325 us, p99 3,421 us, and p99.9
3,939 us. The workload used seeded Poisson arrivals at a mean 1,500 requests/s
and concurrency 128; its histogram includes client admission delay. These
numbers are a single hosted-run observation, not a V0/V1 comparison, production
SLO, or full-model result. See the
[recorded workflow run](https://github.com/pranavkkp4/Distributed-Fraud-Detection/actions/runs/30587168363).

Local RTX 3050 component measurements found captured-stub p50/p99 of
40.5/59.5 us and direct-stub p50/p99 of 40.4/58.1 us. The experiment therefore
does not show a CUDA Graph latency improvement for that tiny workload. Nsight
Systems recorded a small genuine host/GPU overlap window, while Nsight Compute
hardware counters were unavailable under the host's counter permissions. The
manifests and diagnostics are under
[`docs/profiling/results`](../profiling/results/).

No sub-millisecond end-to-end result is committed as fact. Loading the exported
transformer in the native daemon, profiling that model with permitted Nsight
Compute counters, repeating the HTTP workload across controlled machines, and
comparing V0-V5 variants under identical load remain required before promoting
the design targets to measured claims.
