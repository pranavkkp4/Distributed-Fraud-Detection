# Optimization history: V0 to V7

This is the experiment plan and decision log. Values are populated only after a
reproducible run following `docs/profiling/README.md`; blank results are deliberate.

| Version | Change | Hypothesis | Required comparison |
| --- | --- | --- | --- |
| V0 | PyTorch eager FP32 | correctness baseline | score distribution and API latency |
| V1 | compiled PyTorch, fixed shapes | remove framework dispatch | V0 vs V1 warm inference |
| V2 | C++ runtime with vendor FP16 kernels | remove Python and improve layout | V1 vs V2 GPU interval |
| V3 | CUDA Graph replay and stream arena | amortize launches and allocation | uncaptured vs captured p99 |
| V4 | calibrated E4M3 GEMM, FP32 accumulation | halve operand traffic on capable GPUs | vendor FP8 vs custom, plus score drift |
| V5 | short-context fused attention | reduce HBM round trips | standard vs fused by sequence length |
| V6 | bounded continuous batching | gain burst throughput without tail loss | batch off/on at equal offered load |
| V7 | replicated distributed workers | survive failure and scale horizontally | one vs N replicas and injected loss |

## Promotion gates

An optimization advances only when it passes all applicable gates:

- maximum absolute/relative numeric error stays within the declared calibration
  tolerance and decision flips are analyzed;
- API p99 does not regress at the target arrival rate;
- deadline-miss and fallback rates remain inside the error budget;
- the result repeats across at least five measured runs;
- the capability fallback is tested on hardware that lacks the optimized path;
- raw profiler/load artifacts and a completed manifest are archived.

## Why tensor parallelism is not V7

The compact model should be replicated first. Tensor parallelism adds collective
communication and is evaluated only when a model cannot fit on one device or a
measured memory-bound workload benefits from the split. A future experiment must
compare replication and tensor parallelism at the same availability and load; the
word “distributed” alone is not evidence of a speedup.

