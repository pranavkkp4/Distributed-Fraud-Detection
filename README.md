# Distributed Fraud Detection Engine

[![Continuous integration](https://github.com/pranavkkp4/Distributed-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/pranavkkp4/Distributed-Fraud-Detection/actions/workflows/ci.yml)

A runnable, capability-gated reference implementation for latency-sensitive fraud
scoring. It combines a fixed-shape transaction transformer lifecycle, C++/CUDA
kernel primitives, a deadline-aware Go serving plane, Kafka/Redis feature
materialization, and ready-to-run Compose/Minikube observability.

The repository does **not** claim that every deployment is sub-millisecond. That
number is an objective for a warmed, co-located hot path. Any result must include
the hardware, model bundle hash, load shape, phase histograms, and raw profiler
artifacts described in [the profiling protocol](docs/profiling/README.md).

## What is implemented

| Plane | Executable baseline |
| --- | --- |
| Offline model lifecycle | PyTorch 16×32 encoder, deterministic data, E4M3/E5M2 activation calibration, artifact-bound bundle metadata, ONNX and optional TensorRT export |
| Native kernels | FP32 correctness oracle, real CUDA FP16 and capability-gated E4M3 storage/conversion paths with FP32 accumulation, stream-ordered arena, explicit backend/fallback reporting |
| Serving | Authenticated HTTPS API, authenticated TLS Redis reads, phase deadlines, bounded continuous batching, long-lived mTLS gRPC worker pool, health, metrics, deterministic fail-closed responses |
| Streaming | Authenticated TLS Kafka/Redis, bounded retries, idempotent capacity-bounded event-time windows, 32-value materialization, opt-in durable DLQ, in-memory development mode |
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
    client[Authorization client] --> gateway[HTTP gateway]
    gateway --> redis
    gateway --> batcher[Bounded batcher]
    batcher --> workers[gRPC worker replicas]
    workers -. native adapter milestone .-> cuda[C++ / CUDA pipeline]
    trainer[PyTorch + calibration] --> bundle[(Immutable model bundle)]
    bundle -. native adapter milestone .-> workers
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

## Run from source

The minimum source toolchains are Python 3.11, Go 1.24, and CMake 3.20 with a C++17
compiler. CUDA 11.8+ is optional; configuration falls back to the CPU reference when
CUDA is absent.

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

## Validation

The root Makefile provides the common entry points:

```text
make test-python       # model, calibration, integrity, real ONNX export
make test-go           # serving and streaming unit tests
make test-go-race      # Go race detector, including cross-plane contracts
make test-cpp          # CPU CMake build and CTest
make test-integration  # offline cross-plane feature/state contracts
make test-smoke        # harness against a running gateway
make test-api          # Karate against a running gateway
make validate-config   # YAML, JSON, XML, and repository contract checks
```

`tests/integration/smoke_load.py` is a correctness and small-load harness. Its
latency output is observational and never treated as an SLO certification. CI
runs the Python/ONNX, Go race, CPU native, configuration, live gRPC/HTTP, and
Karate suites. CUDA correctness is run where a CUDA runner is available; hosted
CPU CI does not pretend to validate a GPU.

## Repository layout

```text
docs/                 architecture, profiling protocol, optimization history
training/             model, FP8 calibration, ONNX/TensorRT bundle export
execution_engine/     portable C++ and optional CUDA kernel pipeline
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
