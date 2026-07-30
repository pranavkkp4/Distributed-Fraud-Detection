# Cross-plane validation

Run the dependency-free public Go contracts from this directory:

```powershell
go test ./...
```

They establish the fixed-width, version-namespaced feature-key contract and the
stream aggregation offset/idempotency contract. They do not start Redis or Kafka.

Run the HTTP smoke harness against a running gateway:

```powershell
$env:DFD_AUTH_TOKEN = 'local-development-token'
python smoke_load.py --requests 20 --concurrency 4
```

If the gateway is absent it prints `SKIP` and exits successfully. The request
output is intentionally not an SLO or performance certification.

The full external API suite is in `../api` and uses Karate/Maven:

```powershell
mvn -f ../api/pom.xml test -DbaseUrl=http://127.0.0.1:8080 -DauthToken=local-development-token
```

The API suite requires a gateway configured with `FRAUD_AUTH_TOKEN` equal to the
test token; its authentication scenarios intentionally fail against an open
development gateway.
