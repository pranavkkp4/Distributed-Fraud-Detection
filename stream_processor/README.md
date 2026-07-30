# Stream processor

For a local demonstration, supply an array of ordered events on standard input:

```powershell
cd stream_processor
$env:FRAUD_DEVELOPMENT_INSECURE = "true" # local-only plaintext dependencies and probes
'[{"id":"e-1","entity_id":"demo-account","amount":15,"occurred_at":"2026-01-01T00:00:00Z","partition":0,"offset":1}]' | go run ./cmd/stream-processor
```

When `KAFKA_BROKERS` is set, the command uses the real Kafka consumer (`KAFKA_TOPIC` defaults
to `financial-events`; `KAFKA_GROUP_ID` defaults to `fraud-feature-aggregator`). `REDIS_ADDR`
is then mandatory so consumed state cannot disappear into a process-local sink. The stored vector has 32 values: index 0 is rolling count,
index 1 rolling amount sum, index 2 rolling maximum, and indexes 3-31 are reserved zeros.
Kafka records are committed only after successful materialization. Operational endpoints are
served on `STREAM_PROCESSOR_HTTP_ADDR` (default `:8082`). Fetch, processing, and commit retries
are bounded by `KAFKA_MAX_ATTEMPTS` and `KAFKA_RETRY_BACKOFF`; poison records halt by default.
Set `KAFKA_POISON_POLICY=dlq` with a nonempty `KAFKA_DLQ_TOPIC` different from the source topic
to use the built-in synchronous Kafka DLQ writer (`RequireAll` acknowledgement). The original
Kafka key/value are preserved and fixed source topic/partition/offset metadata headers are added;
the source offset is committed only after the DLQ write succeeds. A custom durable handler remains
available to library users.

Outside explicit development-insecure mode, Kafka uses TLS 1.3 and requires
either `KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD` (SCRAM-SHA-512) or
`KAFKA_TLS_CERT_FILE` / `KAFKA_TLS_KEY_FILE`; configure trust with
`KAFKA_TLS_CA_FILE` and optional `KAFKA_TLS_SERVER_NAME`. Redis likewise
requires TLS 1.3 plus a password or client certificate. Operational endpoints
use HTTPS with `STREAM_PROCESSOR_TLS_CERT_FILE` /
`STREAM_PROCESSOR_TLS_KEY_FILE`.

`STREAM_MAX_ENTITIES` and
`STREAM_MAX_EVENTS` bound in-memory state. Reaching event capacity fails closed rather than
silently dropping history and publishing an incorrect window. Run tests with `go test ./...`.
