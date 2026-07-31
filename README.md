# Distributed Fraud Detection Engine

[![Continuous integration](https://github.com/pranavkkp4/Distributed-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/pranavkkp4/Distributed-Fraud-Detection/actions/workflows/ci.yml)

A runnable, capability-gated reference implementation for latency-sensitive fraud
scoring. It combines a fixed-shape transaction transformer lifecycle, C++/CUDA
kernel primitives, a deadline-aware Go serving plane, Kafka/Redis feature
materialization, and ready-to-run Compose/Minikube observability. A second,
Linux-only low-latency path joins an authenticated Rust HTTP gateway to a
no-allocation C++ scheduler through verified FlatBuffers in System V shared
memory.

The repository does **not** claim that every deployment is sub-millisecond. That
number is an objective for a warmed, co-located hot path. Any result must include
the hardware, model bundle hash, load shape, phase histograms, and raw profiler
artifacts described in [the profiling protocol](docs/profiling/README.md).

## What is implemented

| Plane | Executable baseline |
| --- | --- |
| Offline model lifecycle | PyTorch 16×32 encoder, deterministic data, E4M3/E5M2 activation calibration, artifact-bound bundle metadata, ONNX and optional TensorRT export |
| Native kernels | FP32 correctness oracle, real CUDA FP16 and capability-gated E4M3 storage/conversion paths with FP32 accumulation, stream-ordered arena, explicit backend/fallback reporting |
| Shared-memory inference | Mode-0600 System V segment, versioned cross-language ABI, bounded lock-free ring, in-place FRTX verification/decoding, deadline/full-batch scheduler, CUDA Graph lifecycle, and SM80 CUTLASS INT8 GEMM component |
| Rust transaction API | Constant-time bearer authentication, bounded JSON, direct fixed-slot FlatBuffer construction, absolute deadlines, cancellation recovery, and closed 200/422/503 response semantics |
| Serving | Authenticated HTTPS API, authenticated TLS Redis reads, phase deadlines, bounded continuous batching, long-lived mTLS gRPC worker pool, health, metrics, deterministic fail-closed responses |
| Streaming | Authenticated TLS Kafka/Redis, bounded retries, idempotent capacity-bounded event-time windows, 32-value materialization, opt-in durable DLQ, in-memory development mode |
| Load and profiling | Seeded open-loop Poisson arrivals, bounded concurrency, HDR recording, JSON/CSV/SVG reports, Linux `perf` workflow, and checked-in CUDA/Nsight evidence |
| Operations | Multi-stage images, Compose, Minikube, Prometheus alerts, Grafana dashboard, unit/integration/Karate suites |

The local Go worker is intentionally a deterministic CPU scorer. The CUDA code is
a kernel-validation pipeline, not an adapter for the exported full transformer.
Connecting ONNX/TensorRT bundle loading to the native worker is a separate, clearly
tracked milestone. See [implementation status](docs/architecture/implementation_status.md)
for the exact boundary.

## Architecture

```mermaid
flowchart LR
    events[Transaction events] --> kafka[(Kafka)]
    kafka --> stream[Window aggregator]
    stream --> redis[(Redis features)]
    client[Authorization client] --> gateway[Go HTTP gateway]
    gateway --> redis --> batcher[Go bounded batcher]
    batcher --> workers[gRPC worker replicas]
    workers -. native adapter milestone .-> cuda[C++ / CUDA pipeline]
    client --> rust[Rust transaction gateway]
    rust --> shm[FlatBuffer slots + lock-free ring]
    shm --> daemon[C++ SLA batcher]
    daemon --> graphs[CPU reference or CUDA Graph]
    trainer[PyTorch + calibration] --> bundle[(Immutable model bundle)]
    bundle -. native adapter milestone .-> workers
    bundle -. model adapter milestone .-> daemon
```

The synchronous path is one bounded feature read and one inference dispatch. The
streaming plane precomputes rolling state so authorization never performs an
online historical query. Full diagrams and failure semantics are under
[`docs/architecture`](docs/architecture/).

## Quick start with Docker Compose

Prerequisites are Docker with Compose v2. Copy the development configuration,
start the stack, and wait for the gateway to become ready:

```powershell
Copy-Item .env.example .env
docker compose -f infrastructure/compose/docker-compose.yml up --build -d
Invoke-RestMethod http://127.0.0.1:8080/readyz
```

Score the built-in 32-feature demo account:

```powershell
$headers = @{
  Authorization = "Bearer local-development-token"
  "X-Request-ID" = "demo-001"
}
$body = @{entity_id = "demo-account"; amount = 42.0} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/score `
  -Headers $headers -ContentType application/json -Body $body
```

Prometheus and Grafana are exposed for local inspection at
`http://127.0.0.1:9093` and `http://127.0.0.1:3000`. Credentials and tokens in the
example files are development-only. Every published Compose port is bound to
`127.0.0.1`; remote access requires an explicit, reviewed deployment change.

## Shared-memory quick start (Linux)

This path requires a 64-bit little-endian Linux host, CMake 3.24+, a C++20
compiler, Rust 1.81+, and Go 1.26+ for the optional load generator. The daemon
owns the System V segment; start it before the gateway:

```bash
make build-daemon build-rust
export FRAUD_SHM_KEY=1178880336
export FRAUD_GATEWAY_TOKEN=local-shm-token-change-me
build/inference-daemon/fraud_inference_daemon "$FRAUD_SHM_KEY" 1024 4096 &
daemon_pid=$!
FRAUD_GATEWAY_ADDR=127.0.0.1:8081 \
  api_gateway/target/release/fraud-shm-api-gateway &
gateway_pid=$!
```

Send an exact integer-micros transaction:

```bash
curl -sS http://127.0.0.1:8081/v1/transactions \
  -H "Authorization: Bearer $FRAUD_GATEWAY_TOKEN" \
  -H 'content-type: application/json' \
  --data '{"transaction_id":"tx-1","account_id":"acct-1","amount_micros":12340000,"currency":"USD","occurred_at_ns":1735689600000000000}'
kill "$gateway_pid" "$daemon_pid"
```

The current CPU reference decision is deterministic and exists to validate the
wire contract; it is not the exported fraud model. The daemon verifies slot
bounds, the `FRTX` identifier, required fields, and request-ID agreement before
using borrowed FlatBuffer views. Terminate TLS at a trusted local proxy if this
loopback-only gateway is exposed beyond a development host.

## Run from source

The complete repository uses Python 3.11, Go 1.26, Rust 1.81, and CMake 3.24
with a C++20 compiler. CUDA 11.8+ is optional; configuration falls back to the
CPU reference when CUDA is absent.

Install the complete model-export environment and run its tests:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r training/requirements.txt "PyYAML==6.0.3"
.\.venv\Scripts\python -m unittest tests/training/test_training_plane.py -v
```

Run the worker and gateway in separate terminals:

```powershell
cd serving_plane
$env:FRAUD_WORKER_ADDR = ":50051"
$env:FRAUD_WORKER_AUTH_TOKEN = "local-worker-token"
$env:FRAUD_DEVELOPMENT_INSECURE = "true" # local-only plaintext transport
go run ./cmd/worker
```

```powershell
cd serving_plane
$env:FRAUD_WORKER_ADDRS = "127.0.0.1:50051"
$env:FRAUD_WORKER_AUTH_TOKEN = "local-worker-token"
$env:FRAUD_AUTH_TOKEN = "local-development-token"
$env:FRAUD_DEVELOPMENT_INSECURE = "true" # local-only plaintext transport
go run ./cmd/gateway
```

Build and test the portable native path:

```powershell
cmake -S execution_engine -B build/execution-cpu `
  -DFRAUD_ENGINE_ENABLE_CUDA=OFF -DFRAUD_ENGINE_BUILD_TESTS=ON
cmake --build build/execution-cpu --config Release --parallel
ctest --test-dir build/execution-cpu -C Release --output-on-failure
```

Change `FRAUD_ENGINE_ENABLE_CUDA` to `ON` for a CUDA-capable toolchain. Runtime
capability checks still decide the preferred precision, and every result records
the backend that actually executed.

## Model preparation

```powershell
python -m training.cli sample --batch-size 8 --seed 7
python -m training.cli calibrate training/artifacts/calibration.json --format E4M3
python -m training.export.export_model training/artifacts/fraud-model-v1 --version 0.1.0
```

Bundle creation is write-once. `metadata.json` includes the fixed-shape contract,
calibration metadata, SHA-256 for every runtime artifact, and a fingerprint that
commits to those hashes. TensorRT construction is best-effort and never reports an
FP8 engine when `trtexec` or compatible hardware rejects the build.

## Performance evidence and V0–V5 narrative

The latency values from the project brief are comparison targets, not measured
results from this repository:

| Project-brief milestone | Illustrative p99 | Status |
| --- | ---: | --- |
| V0 — naive PyTorch API | 150 ms | Unmeasured here |
| V1 — C++ API over TCP | 45 ms | Unmeasured here |
| V2 — shared-memory IPC | 12 ms | Target only; not a like-for-like V1/V2 comparison |
| V3 — CUDA Graphs and pinned memory | 4 ms | Target only; component timing is not API timing |
| V4 — calibrated low precision | Not specified | Component correctness only |
| V5 — fused short-context attention | Not specified | Reference/microkernel coverage only |

One seeded CPU-only GitHub-hosted integration run at commit `d71b52fd` sent
15,000 open-loop Poisson arrivals at a mean 1,500 requests/s through the local
Rust HTTP/System V/C++ reference path, including C++ FRTX verification and
decoding. It completed all 15,000 with HTTP 200 and no transport errors:

| Observed scope | Throughput | p50 | p90 | p99 | p99.9 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Planned arrival through HTTP completion | 1,505.70/s | 2.325 ms | 3.145 ms | 3.421 ms | 3.939 ms | 4.943 ms |

Five local RTX 3050 component runs found direct batch-32 copy/stub/synchronize
p50/p99 of 40.4/58.1 µs and CUDA Graph replay p50/p99 of 40.5/59.5 µs. Graph
replay therefore did not improve this deliberately tiny launch-dominated stub.
Compute Sanitizer reported zero errors, the CUTLASS INT8 GEMM matched its CPU
oracle, and Nsight Systems recorded a small real host/device overlap window.
Nsight Compute counters were unavailable under `ERR_NVGPUCTRPERM`, so no
occupancy or Tensor Core utilization claim is made.

These results do not load the exported transformer and do not establish a
production SLO or capacity limit. The manifests, latency CDF, real daemon flame
graph, Nsight export, and permission diagnostics are under
[profiling results](docs/profiling/results/). The methodology and cited
10-page paper are available as [source](docs/paper/research_paper.tex) and
[PDF](docs/paper/research_paper.pdf).

## Validation

The root Makefile provides the common entry points:

```text
make test-python       # model, calibration, integrity, real ONNX export
make test-go           # serving and streaming unit tests
make test-go-race      # Go race detector, including cross-plane contracts
make test-cpp          # CPU CMake build and CTest
make test-daemon       # arena, lock-free ring, batcher, FRTX verification
make test-rust         # Rust formatting, clippy, and gateway tests
make test-load         # Poisson/HDR generator format, vet, and tests
make verify-flatbuffers # regenerate and compare Rust/C++ schema bindings
make test-ipc          # live authenticated Linux HTTP/System V integration
make test-integration  # offline cross-plane feature/state contracts
make test-smoke        # harness against a running gateway
make test-api          # Karate against a running gateway
make validate-config   # YAML, JSON, XML, and repository contract checks
make paper             # regenerate measured figures and compile the cited PDF
```

`tests/integration/smoke_load.py` is a correctness and small-load harness. Its
latency output is observational and never treated as an SLO certification. CI
runs the Python/ONNX, Go race, both CPU-native paths, Rust lint/tests,
FlatBuffers generation, live authenticated shared-memory IPC, configuration,
gRPC/HTTP, and Karate suites. CUDA correctness is run where a CUDA runner is
available; hosted CPU CI does not pretend to validate a GPU.

## Repository layout

```text
docs/                 architecture, profiling protocol, optimization history
training/             model, FP8 calibration, ONNX/TensorRT bundle export
execution_engine/     portable C++ and optional CUDA kernel pipeline
inference_daemon/      no-GC arena, shared ring, SLA batcher, CUDA Graph/CUTLASS
api_gateway/           Linux Rust HTTP-to-System-V gateway
schemas/               shared FlatBuffers Transaction contract
load_tester/           seeded Poisson/HDR high-concurrency load generator
serving_plane/        scheduler, feature store, gRPC workers, HTTP gateway
stream_processor/     Kafka ingestion and sliding-window materialization
infrastructure/       Dockerfiles, Compose, Minikube, Prometheus, Grafana
tests/                native, training, integration, and Karate API coverage
```

## Safety and measurement rules

- Callers and all child operations have bounded deadlines.
- Unknown features or exhausted worker deadlines produce an explicit degraded,
  fail-closed decision; they never silently allow a transaction.
- Authentication is mandatory in supplied deployment examples and is propagated
  to workers through long-lived internal channels.
- Production startup rejects plaintext data-plane transports: gateway, worker,
  stream telemetry, Redis, and Kafka use TLS 1.3, with Redis ACL/mTLS and Kafka
  SCRAM-SHA-512/mTLS identity options.
- Native low-precision results are compared with the FP32 oracle. CPU fallback is
  labeled as CPU and includes a machine-readable reason.
- FP8 is never selected solely because the code compiled; device capability,
  calibration eligibility, and runtime execution must all succeed.
- No benchmark table is populated without its manifest and raw artifacts.
