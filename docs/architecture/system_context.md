# System context

The fraud inference engine turns an authorized transaction request into an allow,
review, or decline recommendation. Its sub-millisecond objective applies to the
warm scoring path inside one region; event ingestion, model training, deployment,
and replay are explicitly outside that latency boundary.

```mermaid
flowchart LR
    client[Payment authorization system]
    engine[Fraud inference engine]
    analyst[Fraud analyst]
    registry[Immutable model registry]
    events[(Financial event stream)]

    client -->|score transaction| engine
    engine -->|decision + score + confidence| client
    events -->|ordered events by entity| engine
    analyst -->|policy and threshold changes| engine
    registry -->|versioned model bundle| engine
    engine -->|decision/audit event| events
```

## Target production guarantees

These are the contracts the production deployment is designed to enforce. The
reference implementation's exact coverage and known gaps are tracked in
[`implementation_status.md`](implementation_status.md).

- Ordering is guaranteed per chosen entity key, not globally.
- The synchronous path performs one fixed-width feature lookup and one worker
  dispatch at most.
- Every internal operation receives a deadline no later than the caller's.
- A timeout or unhealthy worker activates a deterministic, auditable fallback.
- Native FP8 is selected only when the compiled backend and device capability
  support it; FP16 and CPU reference paths remain available.
- Scores and explanations identify the exact model, calibration, and policy
  versions that produced them.

The project does not promise universal sub-millisecond latency. It supplies the
instrumentation and benchmark protocol needed to demonstrate that property on a
named hardware, network, data, and load configuration.
