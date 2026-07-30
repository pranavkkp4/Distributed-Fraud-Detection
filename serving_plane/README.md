# Serving plane

Run a local deterministic worker and gateway in separate terminals:

```powershell
cd serving_plane
$env:FRAUD_WORKER_AUTH_TOKEN = "local-worker-token"
$env:FRAUD_DEVELOPMENT_INSECURE = "true" # local-only plaintext gRPC
go run ./cmd/worker
```

```powershell
cd serving_plane
$env:FRAUD_WORKER_ADDR = "127.0.0.1:50051"
$env:FRAUD_WORKER_AUTH_TOKEN = "local-worker-token"
$env:FRAUD_DEVELOPMENT_INSECURE = "true" # local-only plaintext gRPC
go run ./cmd/gateway
```

`POST /v1/score` accepts `{"entity_id":"demo-account","amount":42}`.
Set `FRAUD_AUTH_TOKEN` to require a matching Bearer token. `REDIS_ADDR` enables
single-shot Redis feature reads (`entity_id` values are comma-separated fixed-width vectors);
the built-in `demo-account` snapshot remains a development-only local fallback. Production
startup requires Redis and reports Redis loss as not ready. Health, readiness,
and OpenMetrics-style metrics are available at `/healthz`, `/readyz`, and `/metrics`.
The gateway expects the model contract's 32-value feature vector. Worker operational
health/readiness/metrics are served on `FRAUD_WORKER_HTTP_ADDR` (default `:9091`), while
gRPC uses `FRAUD_WORKER_ADDR` (default `:50051`). Score responses include bounded confidence,
up to three clipped feature-contribution explanations (indices only; never raw features), and
model/calibration/policy/feature version identifiers.

For replicas, set comma-separated `FRAUD_WORKER_ADDRS`; the singular variable remains an alias.
Deadline phases use `FRAUD_TIMEOUT`, `FRAUD_FEATURE_TIMEOUT`, `FRAUD_QUEUE_DELAY`, and
`FRAUD_INFERENCE_TIMEOUT` (defaults 40ms, 10ms, 2ms, and 20ms). Fail-closed responses carry
an enum-like `fallback_reason` without internal error text. Both binaries support one-shot
`-healthcheck` mode for container probes. In development mode it calls local HTTP. In
production it calls local HTTPS and verifies the certificate against platform roots or
`FRAUD_GATEWAY_TLS_CA_FILE` / `FRAUD_WORKER_HTTP_TLS_CA_FILE`; use the matching
`*_TLS_SERVER_NAME` when the certificate does not include `127.0.0.1`.

Production startup rejects missing authentication tokens and plaintext gRPC.
Set worker certificate/key paths with `FRAUD_WORKER_TLS_CERT_FILE` and
`FRAUD_WORKER_TLS_KEY_FILE`; gateways trust the worker CA through
`FRAUD_WORKER_TLS_CA_FILE` and present their internal identity through
`FRAUD_GATEWAY_MTLS_CERT_FILE` / `FRAUD_GATEWAY_MTLS_KEY_FILE`. Workers validate
that identity with `FRAUD_GATEWAY_MTLS_CA_FILE`. The public HTTPS listener uses
the separate `FRAUD_GATEWAY_TLS_CERT_FILE` / `FRAUD_GATEWAY_TLS_KEY_FILE` pair,
and worker operational HTTPS uses `FRAUD_WORKER_HTTP_TLS_CERT_FILE` /
`FRAUD_WORKER_HTTP_TLS_KEY_FILE`. Plaintext is only permitted with the explicit
`FRAUD_DEVELOPMENT_INSECURE=true` local flag.

Run the suite with `go test ./...` (and `go test -race ./...` in CI).
