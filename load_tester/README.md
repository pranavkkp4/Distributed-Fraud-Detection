# Load tester

`go run . -url http://127.0.0.1:8080/v1/score -token dev-token -requests 10000 -concurrency 500 -rate 2000`

The scheduler uses seeded exponential inter-arrival times (a Poisson process), while `-concurrency` bounds in-flight sockets and goroutines. `-duration` may replace `-requests`; its request count is calculated from the configured mean rate. Entity IDs support deterministic weighted distribution, for example `-entities "vip:8,regular:2"` sends eight VIP records for every two regular records.

Every completed HTTP request—including transport failures—is recorded in an HDR histogram. Latency begins at the planned open-loop arrival, rather than worker admission, so it includes client-side queue delay under saturation. Because these are already measured planned-arrival observations, the tester uses `RecordValue`, not synthetic coordinated-omission correction. Non-2xx responses and transport failures count as errors, but retain distinct status counts. The output directory contains a JSON manifest (including runtime environment), quantile CSV, and a measured self-contained latency CDF; these are run artifacts, not benchmark claims.

Use HTTPS in non-development environments. The tool does not load client certificates or bypass TLS verification.

## Payload modes

`-payload-mode entity` is the default and sends the existing gateway body: `{"entity_id":"…","amount":10}`. Use `-payload-mode transaction` for the Rust Transaction API; it sends `transaction_id`, `account_id`, `amount_micros`, `currency`, `occurred_at_ns`, and `merchant_category`. Transaction IDs are deterministic and unique per seed/index (`load-<seed>-<index>`); account IDs follow the configured weighted entity mix.
