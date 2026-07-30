# Profiling protocol

No performance number belongs in this repository without the command, commit,
hardware, software versions, load shape, and raw artifact that produced it.

## Three latency scopes

1. **Kernel latency**: CUDA events around an individual kernel, after warmup.
2. **GPU interval latency**: packing through graph completion, including copies.
3. **API decision latency**: client send through response receipt, split into
   parsing, feature lookup, queueing, worker, and response phases.

Report p50, p95, p99, p99.9, maximum, throughput, error rate, and fallback rate.
Keep cold and warm runs separate. Use a fixed seed and archive the exact request
corpus or its generator configuration.

## Kernel workflow

```text
ncu --set full --import-source yes --target-processes all \
  --export docs/profiling/results/<commit>-<gpu>-kernel \
  <execution-engine-benchmark>
```

Capture achieved occupancy, tensor-core utilization, DRAM bytes, L2 hit rate,
register pressure, shared-memory use, and dominant warp stall reasons. Compare the
FP32 reference, vendor FP16/FP8 path, and custom path on identical shapes.

## End-to-end workflow

Warm the feature keys and graph variants before measurement. Drive a documented
arrival distribution rather than only closed-loop maximum throughput. Record
queue-delay caps and repeat with a worker removed and Redis delayed. Use
`benchmark_manifest.example.json` as the machine-readable record.

