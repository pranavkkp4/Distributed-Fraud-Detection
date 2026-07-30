# Implementation status

The repository separates executable behavior from measured optimization claims.
“Implemented” below means source plus automated correctness coverage; it does not
mean a latency target has been demonstrated on arbitrary hardware.

| Area | Implemented baseline | Deliberate boundary |
| --- | --- | --- |
| Model lifecycle | Fixed-shape 16×32 transformer, deterministic samples, activation calibration, immutable artifact hashes, ONNX export, optional TensorRT build | Training data and model quality are synthetic; a production registry/signing adapter is not included |
| CPU execution | FP32 projection and short-context attention microkernel reference with finite-input validation | It is a component-level correctness oracle, not the full exported model or latency target |
| CUDA execution | Capability selection, real FP16 storage/conversion microkernels with FP32 accumulation, SM 8.9+ E4M3 path, stream-ordered arena, explicit CPU fallback reporting | The pipeline cannot load the exported transformer and makes no Tensor Core throughput claim; CUDA Graph capture reports unavailable until stable-buffer replay is implemented |
| Serving | Authenticated HTTPS gateway, authenticated TLS 1.3 Redis, phase deadlines, bounded continuous batching, mTLS gRPC workers, replica health/circuit routing, deterministic fail-closed response, health and metrics | The local worker is a deterministic scorer; production requires Redis, and the C++/TensorRT worker adapter remains a separate integration milestone |
| Streaming | Authenticated TLS 1.3 Kafka/Redis transports, partition-order consumption with bounded fetch/process/commit retries, idempotent event-time windows, capacity-bounded state, fixed-width materialization, local mode | Capacity exhaustion halts safely; poison records halt by default or use an explicitly configured synchronous DLQ, while exactly-once deployment still requires Kafka transactions and durable state |
| Deployment | Compose topology, Minikube manifests, Prometheus alerts, Grafana dashboard, native and API regression suites | Docker/Kubernetes files require their external runtimes and do not prove production capacity; local image tags must be replaced with registry digests during controlled promotion |

## Numeric backend terminology

`preferred_backend` describes what device capability permits. `backend` on an
inference result describes what actually produced the result. A failed CUDA path
is recomputed by the FP32 reference and is reported as CPU with `fallback_used`;
the result is never labeled FP16 or FP8 merely because a compatible device was
detected.

The current E4M3 CUDA path stores operands in FP8 and accumulates scalar products
in FP32. It is useful for accuracy and integration validation, but it is not a
Tensor Core GEMM and the repository makes no throughput claim for it. CUTLASS or
cuBLASLt comparison and a tiled Tensor Core kernel belong to a measured V4
optimization, not to the baseline label.

## Performance status

No sub-millisecond result is committed as a fact. The files under
`docs/profiling/` define the evidence required: commit, model bundle hash,
hardware/software inventory, load distribution, cold/warm separation, phase
histograms, failure rate, and raw artifacts. Until a completed manifest and raw
trace are checked in, 1 ms is an objective rather than a benchmark result.
