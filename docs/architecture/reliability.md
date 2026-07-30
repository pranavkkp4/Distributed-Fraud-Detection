# Target reliability and failure semantics

This document defines the desired production behavior. It is a contract and
review checklist, not a claim that every item is benchmarked or connected in the
local demo. Current coverage is listed in
[`implementation_status.md`](implementation_status.md).

## Deadline propagation

The gateway derives one absolute deadline from the earlier of the client deadline
and its configured request timeout. Feature retrieval, queueing, and worker calls
receive shrinking child budgets. A component must never start work after the
request is already expired.

## Fallback policy

The implemented fallback is deterministic and bounded. When feature retrieval
succeeds but worker inference fails with enough deadline remaining, it evaluates
the retrieved fixed-width feature block, returns a degraded flag and
machine-readable reason, and increments a labeled fallback counter. If feature
retrieval itself fails, production mode fails closed because no trustworthy
feature block is available. Synthetic features are available only in the explicit
insecure development mode.

A manual-review route and a durable last-known-good feature cache are target
capabilities, not current behavior. Silent allow-on-error remains forbidden.

## Delivery and state

Streaming events use an entity identifier as the partition key. Aggregation keeps
a high-water mark and event IDs per entity, making redelivery idempotent. Feature
state updates are staged in memory and committed only after validation succeeds.
The current Redis materializer performs one fixed-width `SET`; it does not yet use
compare-and-set version fencing, so multiple stream-processor writers for the same
entity are unsupported. Kafka transactions, durable aggregate state, and Redis
version fencing are required before advertising exactly-once multi-writer state.

## Worker health and rollout

- Readiness is false until a model bundle is validated and warmed.
- Consecutive failures open a worker circuit for a bounded cooldown.
- Requests are routed only to ready replicas; retries occur only when remaining
  deadline and idempotency permit them.
- Model rollouts are immutable and versioned. Readiness changes only after warmup,
  enabling surge-first replacement and instant rollback.
- Primary and fallback decisions preserve request ID, feature version, model
  version, calibration version, and policy version for audit.
