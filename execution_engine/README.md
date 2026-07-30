# Fraud microkernel pipeline

This directory currently contains a fixed-shape projection and single-head attention
microbenchmark. It validates CPU, FP16 CUDA, and calibrated E4M3 CUDA kernel behavior,
memory lifecycles, and fallback reporting for a 32-feature, 64-channel, 16-token shape.

It is **not** the exported fraud transformer runtime. It does not load ONNX or TensorRT
artifacts, embeddings, transformer blocks, layer normalization, or the trained classifier
head. Its anomaly score is a microbenchmark diagnostic and must not be represented as
full-model inference or accuracy parity. Integrating the exported model requires a separate
adapter with explicit model metadata validation and end-to-end parity tests.
