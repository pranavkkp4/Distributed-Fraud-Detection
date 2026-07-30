#!/usr/bin/env bash
set -Eeuo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
daemon_bin="${FRAUD_DAEMON_BIN:-$repository_root/build/inference-daemon/fraud_inference_daemon}"
gateway_bin="${FRAUD_GATEWAY_BIN:-$repository_root/api_gateway/target/release/fraud-shm-api-gateway}"
gateway_addr="${FRAUD_GATEWAY_ADDR:-127.0.0.1:18081}"
gateway_token="${FRAUD_GATEWAY_TOKEN:-ci-ipc-token-2026}"
ipc_key="${FRAUD_SHM_KEY:-$((0x46000000 | ($$ & 0x00ffffff)))}"
daemon_log="${RUNNER_TEMP:-/tmp}/fraud-ipc-daemon.$$.log"
gateway_log="${RUNNER_TEMP:-/tmp}/fraud-ipc-gateway.$$.log"
daemon_pid=''
gateway_pid=''
owns_segment=false

if [[ "$(uname -s)" != Linux ]]; then
  echo 'System V IPC integration requires Linux.' >&2
  exit 2
fi
for executable in "$daemon_bin" "$gateway_bin"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing executable: $executable" >&2
    exit 2
  fi
done

segment_exists() {
  python3 - "$ipc_key" <<'PY'
import ctypes
import sys
key = int(sys.argv[1], 0)
libc = ctypes.CDLL(None, use_errno=True)
libc.shmget.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_int)
libc.shmget.restype = ctypes.c_int
raise SystemExit(0 if libc.shmget(key, 0, 0) >= 0 else 1)
PY
}

request_shutdown() {
  python3 - "$ipc_key" <<'PY'
import ctypes
import sys
key = int(sys.argv[1], 0)
libc = ctypes.CDLL(None, use_errno=True)
libc.shmget.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_int)
libc.shmget.restype = ctypes.c_int
libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
libc.shmat.restype = ctypes.c_void_p
libc.shmdt.argtypes = (ctypes.c_void_p,)
shmid = libc.shmget(key, 0, 0)
if shmid < 0:
    raise SystemExit(0)
address = libc.shmat(shmid, None, 0)
if address == ctypes.c_void_p(-1).value:
    raise SystemExit(1)
ctypes.c_uint32.from_address(address + 256).value = 1
libc.shmdt(address)
PY
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    kill -INT "$gateway_pid" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 "$gateway_pid" 2>/dev/null || break
      sleep 0.02
    done
    if kill -0 "$gateway_pid" 2>/dev/null; then kill -TERM "$gateway_pid" 2>/dev/null || true; fi
    wait "$gateway_pid" 2>/dev/null || true
  fi
  if [[ "$owns_segment" == true ]]; then
    request_shutdown || true
  fi
  if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
    for _ in {1..50}; do
      kill -0 "$daemon_pid" 2>/dev/null || break
      sleep 0.02
    done
    if kill -0 "$daemon_pid" 2>/dev/null; then kill -TERM "$daemon_pid" 2>/dev/null || true; fi
    wait "$daemon_pid" 2>/dev/null || true
  fi
  if [[ "$owns_segment" == true ]] && segment_exists; then
    ipcrm -M "$ipc_key" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    echo '--- inference daemon log ---' >&2
    cat "$daemon_log" >&2 || true
    echo '--- Rust gateway log ---' >&2
    cat "$gateway_log" >&2 || true
  fi
  rm -f "$daemon_log" "$gateway_log"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if segment_exists; then
  echo "Refusing to reuse existing System V key $ipc_key" >&2
  exit 2
fi

"$daemon_bin" "$ipc_key" 64 4096 >"$daemon_log" 2>&1 &
daemon_pid=$!
for _ in {1..100}; do
  if segment_exists; then owns_segment=true; break; fi
  kill -0 "$daemon_pid" 2>/dev/null || break
  sleep 0.02
done
if [[ "$owns_segment" != true ]]; then
  echo 'Inference daemon did not create its shared-memory segment.' >&2
  exit 1
fi

FRAUD_SHM_KEY="$ipc_key" FRAUD_GATEWAY_TOKEN="$gateway_token" \
FRAUD_GATEWAY_ADDR="$gateway_addr" FRAUD_GATEWAY_DEADLINE_MS=100 \
  "$gateway_bin" >"$gateway_log" 2>&1 &
gateway_pid=$!

base_url="http://$gateway_addr"
ready=false
for _ in {1..100}; do
  if curl --fail --silent --show-error "$base_url/health" >/dev/null; then ready=true; break; fi
  kill -0 "$gateway_pid" 2>/dev/null || break
  sleep 0.05
done
if [[ "$ready" != true ]]; then
  echo 'Rust gateway did not become healthy.' >&2
  exit 1
fi

payload='{"transaction_id":"ci-ipc-1","account_id":"acct-ci","amount_micros":12340000,"currency":"USD","occurred_at_ns":1735689600000000000,"merchant_category":"retail"}'
unauthorized_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  -H 'content-type: application/json' --data "$payload" "$base_url/v1/transactions")"
[[ "$unauthorized_status" == 401 ]]

for request_number in 1 2 3; do
  response="$(curl --fail --silent --show-error \
    -H "Authorization: Bearer $gateway_token" -H 'content-type: application/json' \
    --data "${payload/ci-ipc-1/ci-ipc-$request_number}" "$base_url/v1/transactions")"
  python3 - "$response" <<'PY'
import json
import math
import sys
body = json.loads(sys.argv[1])
assert body["status"] == 200, body
assert body["decision"] in (0, 1), body
assert math.isfinite(body["score"]), body
assert body["request_id"] > 0 and body["completed_ns"] > 0, body
PY
done

kill -0 "$daemon_pid"
kill -0 "$gateway_pid"
echo 'System V IPC HTTP integration passed (authentication and three transactions).'
