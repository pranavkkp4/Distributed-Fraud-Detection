# Inference daemon

`fraud_inference_daemon` is the native, Linux System V shared-memory endpoint
for the Rust gateway. Its wire ABI is fixed in `include/fraud_daemon/shared_abi.hpp`:
magic `0x46444950`, version `1`, 320-byte header, and 64-byte slot metadata.
It does **not** serialize C++ `std::atomic` objects into shared memory; the ABI
uses aligned integer storage plus compiler/Interlocked atomics.

## CPU build and tests

```sh
cmake -S inference_daemon -B build/daemon -DBUILD_TESTING=ON
cmake --build build/daemon --config Release
ctest --test-dir build/daemon -C Release --output-on-failure
```

The unit tests exercise ABI offsets, a bounded multi-producer/multi-consumer
lock-free ring, deadline and full-batch triggers, dynamic batch sizes, and a
global-allocation-counter assertion around `poll_once()`.

## Linux daemon

```sh
./fraud_inference_daemon 0x46444950 1024 4096
```

The daemon creates a mode-0600 System V segment, initializes all slot sequence
values to their slot index, and removes it after a clean shutdown (`shutdown`
at ABI offset 256). A producer owns a completed response at sequence `p + 2`
until it reads it and calls `release`, which stores `p + capacity`; capacity is
therefore required to be a power of two and at least four. Windows supports the CPU library/tests but intentionally
reports System V IPC as unavailable.

## Optional CUDA / CUTLASS

```sh
cmake -S inference_daemon -B build/daemon-cuda -DFRAUD_DAEMON_ENABLE_CUDA=ON \
  -DFRAUD_DAEMON_ENABLE_CUTLASS=ON -DCUTLASS_PATH=/path/to/cutlass
```

`src/cuda_graph.cu` captures pinned-host H2D, a forward stub, and D2H inside
`cudaStreamBeginCapture`/`cudaStreamEndCapture`, instantiates `cudaGraphExec_t`,
and replays the graph for a captured batch size. `src/cutlass_int8_gemm.cu`
contains an SM80 TensorOp INT8→INT32 wrapper with 128x128x64 threadblock,
64x64x64 warp, and 16x8x32 instruction tiles. The CPU build reports those
capabilities as unavailable instead of advertising an unbuilt accelerator.

At daemon startup, every dynamic batch size from 1 through 32 is pre-captured,
with persistent nonblocking streams and buffers. Replay has no mutex, dynamic
allocation, or stream creation; `shutdown_inference_graphs()` destroys these
resources after scheduling stops.

The CUDA graph currently captures a reference forward stub. Connecting it to
the exported model’s native TensorRT/CUDA execution path is a separate model
integration task; no latency or zero-allocation claim should be inferred for
that external inference implementation.
