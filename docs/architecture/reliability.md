# Reliability and failure semantics

## Deadline propagation

The gateway derives one absolute deadline from the earlier of the client deadline
and its configured request timeout. Feature retrieval, queueing, and worker calls
receive shrinking child budgets. A component must never start work after the
request is already expired.

## Fallback policy

Fallback is intentionally deterministic and bounded. It uses request-local fields
and the last materialized feature block when available. It returns a degraded flag
and machine-readable reason, and increments a labeled fallback counter. Deployers
choose fail-closed or manual-review thresholds; silent allow-on-error is forbidden.

## Delivery and state

Streaming events use an entity identifier as the partition key. Aggregation keeps
a high-water mark and event IDs per entity, making redelivery idempotent. Feature
updates are versioned; a stale write cannot replace a newer state. The local
implementation exposes these semantics through interfaces so Kafka transactions
and Redis compare-and-set operations can be wired without changing the model.

## Worker health and rollout

- Readiness is false until a model bundle is validated and warmed.
- Consecutive failures open a worker circuit for a bounded cooldown.
- Requests are routed only to ready replicas; retries occur only when remaining
  deadline and idempotency permit them.
- Model rollouts are immutable and versioned. Readiness changes only after warmup,
  enabling surge-first replacement and instant rollback.
- Primary and fallback decisions preserve request ID, feature version, model
  version, calibration version, and policy version for audit.

