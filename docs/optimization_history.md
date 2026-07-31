# Optimization history and evidence log

The V0-V7 sequence is a design roadmap, not a table of results already achieved
by this repository. A version becomes historical evidence only after its exact
implementation is measured under the promotion gates below. In particular, the
illustrative V0-V3 figures from the original project brief (150 ms, 45 ms,
12 ms, and 4 ms p99) are targets, not measurements, and are not used as
baselines in this document.

| Project-brief milestone | Illustrative p99 | Evidence status |
| --- | ---: | --- |
| V0 - naive PyTorch API | 150 ms | Unmeasured in this repository |
| V1 - C++ API over TCP | 45 ms | Unmeasured; this exact server is not the implemented native path |
| V2 - shared-memory IPC | 12 ms | Target only; the measured CPU-stub path below observed a different result under a defined workload |
| V3 - CUDA Graphs and pinned memory | 4 ms | Target only; local component measurements below are microseconds but exclude HTTP and model execution |
| V4 | Not specified | A definition and like-for-like protocol must be frozen before measurement |
| V5 | Not specified | A definition and like-for-like protocol must be frozen before measurement |

The repository also retains a more granular engineering roadmap:

| Version | Planned change | Hypothesis | Current evidence state |
| --- | --- | --- | --- |
| V0 | PyTorch eager FP32 | establish correctness and API baseline | Model code exists, but no archived V0 HTTP benchmark exists |
| V1 | compiled PyTorch, fixed shapes | remove framework dispatch | Design target; no like-for-like V0/V1 measurement |
| V2 | native runtime and IPC | remove Python and transport overhead | Linux Rust/FlatBuffers/System V IPC path is implemented; the measured CPU path uses a deterministic stub, not the exported model |
| V3 | CUDA Graph replay and startup buffers | amortize launches and allocation | Component implemented and measured with a one-element FP32 forward stub; it did not improve measured p50 or p99 over direct launch |
| V4 | calibrated low-precision GEMM | reduce operand traffic on capable GPUs | CUTLASS SM80 INT8-to-INT32 component matches its CPU oracle; it is not wired to the transformer and has no end-to-end throughput result |
| V5 | short-context fused attention | reduce HBM round trips | Reference/microkernel coverage exists in `execution_engine`; no standard-versus-fused production-model benchmark exists |
| V6 | SLA-aware continuous batching | gain burst throughput without tail loss | Deadline/full-batch scheduler and variable batch sizes are implemented; the current HTTP run is not a batching ablation |
| V7 | replicated distributed workers | survive failure and scale horizontally | Go worker routing exists separately; no one-versus-N shared-memory worker comparison exists |

## Measured CPU-only shared-memory HTTP path

The [GitHub Actions profiling run](https://github.com/pranavkkp4/Distributed-Fraud-Detection/actions/runs/30587168363)
at commit `d71b52fd17c8e500d2a67901031081fc492e6917` exercised this path on
an Ubuntu 24.04 hosted runner with four logical CPUs:

`Go load tester -> authenticated HTTP -> Rust JSON validation -> FlatBuffers in a shared slot -> System V ring -> C++ FRTX verification/decoding -> SLA batcher -> CPU reference response`

The seeded workload scheduled 15,000 transaction requests with a Poisson mean
of 1,500 arrivals/s, concurrency 128, and seed 20260730. The HDR histogram starts
at each planned arrival, so it includes client-side admission delay as well as
HTTP, serialization, IPC, queueing, and daemon response time.

| Result | Observed value |
| --- | ---: |
| Successful responses | 15,000 / 15,000 |
| Error rate | 0% |
| Observed throughput | 1,505.70 responses/s |
| p50 | 2,325 us |
| p90 | 3,145 us |
| p99 | 3,421 us |
| p99.9 | 3,939 us |
| Maximum | 4,943 us |

This is one reproducible hosted-run observation, not a universal capacity claim
and not a V0-to-V2 speedup measurement. The daemon returned a deterministic CPU
stub decision; it did not load the exported transformer. The workflow also
captured a `perf` flame graph, but a sampled stack profile does not by itself
prove that every daemon path is allocation-free. The narrower scheduler
`poll_once()` no-allocation contract is enforced by a unit-test allocation
counter.

## Measured local CUDA components

Commit `2627824c5bb69773365e183f6e09686642c488f6` was measured on an
NVIDIA GeForce RTX 3050 6 GiB (compute capability 8.6), CUDA 13.3.73,
driver 595.79, CUTLASS 4.6.1, and Windows 10. Each aggregate below is the
median summary from five processes, each with 100 warmups and 2,000 measured
batch-32 iterations.

| Component wall time | p50 | p90 | p99 | p99.9 |
| --- | ---: | ---: | ---: | ---: |
| Captured CUDA Graph | 40.5 us | 43.3 us | 59.5 us | 264.2 us |
| Direct H2D/kernel/D2H | 40.4 us | 42.1 us | 58.1 us | 265.5 us |

These are host wall times including stream synchronization for a tiny FP32
forward stub, not HTTP latency and not transformer inference. Graph replay did
not improve p50 or p99 in this experiment. The CUTLASS 128x128x64 INT8 GEMM
matched the CPU INT32 oracle, and compute-sanitizer reported zero errors. See
the [graph manifest](profiling/results/cuda_graph_replay_2627824.json),
[direct manifest](profiling/results/cuda_direct_2627824.json), and
[compute-sanitizer record](profiling/results/compute_sanitizer_2627824.txt).

The [Nsight Systems summary](profiling/results/nsys_summary_2627824.json)
records a small positive host/GPU overlap window in all 500 probes and async
copy API return before GPU copy completion. This demonstrates execution
overlap mechanics, not a material end-to-end speedup. Nsight Compute could not
collect hardware counters because the host denied performance-counter access;
occupancy, Tensor Core utilization, DRAM/L2 traffic, registers, and warp stalls
remain unmeasured, as recorded in
[the permission diagnostic](profiling/results/ncu_counter_permission_2627824.txt).

## Promotion gates

An optimization advances only when it passes all applicable gates:

- maximum absolute/relative numeric error stays within the declared calibration
  tolerance and decision flips are analyzed;
- API p99 does not regress at the target arrival rate;
- deadline-miss and fallback rates remain inside the error budget;
- the result repeats across at least five measured runs;
- the capability fallback is tested on hardware that lacks the optimized path;
- raw profiler/load artifacts and a completed manifest are archived.

The hosted CPU observation has one run and therefore does not pass the
five-run promotion gate. The CUDA component comparison has five runs, but its
stub workload does not qualify as a full-model or API promotion.

## Why tensor parallelism is not V7

The compact model should be replicated first. Tensor parallelism adds collective
communication and is evaluated only when a model cannot fit on one device or a
measured memory-bound workload benefits from the split. A future experiment must
compare replication and tensor parallelism at the same availability and load;
the word "distributed" alone is not evidence of a speedup.
