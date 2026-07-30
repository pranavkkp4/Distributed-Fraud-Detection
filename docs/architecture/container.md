# Container architecture

```mermaid
flowchart TB
    subgraph control[Streaming control plane]
      kafka[(Kafka / Redpanda)] --> ingest[Ordered ingestion]
      ingest --> agg[Sliding-window aggregator]
      agg --> redis[(Redis feature store)]
    end

    subgraph serving[Serving and memory plane]
      gateway[HTTP gateway] --> scheduler[Deadline-aware batch scheduler]
      scheduler --> feature[Single-shot feature reader]
      feature --> redis
      scheduler --> pool[Worker pool / circuit breaker]
      pool --> workerA[Inference worker A]
      pool --> workerB[Inference worker B]
      gateway --> fallback[Deterministic rules fallback]
    end

    subgraph hardware[Hardware execution plane]
      workerA --> graphA[CUDA Graph cache]
      graphA --> arenaA[Stream-ordered arena]
      graphA --> kernelsA[FP8/FP16 kernels]
      workerB --> graphB[CUDA Graph cache]
      graphB --> arenaB[Stream-ordered arena]
      graphB --> kernelsB[FP8/FP16 kernels]
    end

    subgraph offline[Offline model lifecycle]
      train[Compact transformer training] --> calibrate[FP8 calibration]
      calibrate --> export[ONNX / TensorRT export]
      export --> bundle[(Immutable bundle)]
      bundle --> workerA
      bundle --> workerB
    end

    metrics[Prometheus] -.scrape.-> gateway
    metrics -.scrape.-> workerA
    metrics -.scrape.-> ingest
    grafana[Grafana] --> metrics
```

## Responsibility boundaries

| Plane | Owns | Must not do |
| --- | --- | --- |
| Training | fit, calibrate, validate, sign/export | mutate a live worker model in place |
| Execution | buffers, kernels, graph replay, capability choice | fetch remote features |
| Serving | authentication, deadlines, batching, routing, fallback | calculate historical windows on request |
| Streaming | ordering, deduplication, windows, materialization | synchronously block an authorization |
| Infrastructure | placement, rollout, telemetry, secret references | embed credentials in images or manifests |

