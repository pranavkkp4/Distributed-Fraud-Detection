#!/usr/bin/env bash
# Capture a reproducible System V IPC workload and, where the host permits it,
# perf stacks for the native inference daemon. Missing perf privileges are an
# explicit diagnostic outcome; measured HTTP artifacts are still retained.
set -Eeuo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
daemon_bin="${FRAUD_DAEMON_BIN:-$repository_root/build/profile/inference-daemon/fraud_inference_daemon}"
gateway_bin="${FRAUD_GATEWAY_BIN:-$repository_root/api_gateway/target/release/fraud-shm-api-gateway}"
load_bin="${FRAUD_LOAD_BIN:-$repository_root/build/profile/fraud-load-tester}"
flamegraph_dir="${FLAMEGRAPH_DIR:-$repository_root/build/profile/FlameGraph}"
artifact_dir="${PROFILE_ARTIFACT_DIR:-$repository_root/profile-artifacts}"
gateway_addr="${FRAUD_GATEWAY_ADDR:-127.0.0.1:18081}"
gateway_token="${FRAUD_GATEWAY_TOKEN:-profile-token-2026}"
ipc_key="${FRAUD_SHM_KEY:-$((0x47000000 | ($$ & 0x00ffffff)))}"
request_count="${PROFILE_REQUESTS:-5000}"
arrival_rate="${PROFILE_RATE:-1500}"
concurrency="${PROFILE_CONCURRENCY:-64}"
seed="${PROFILE_SEED:-20260730}"
profile_seconds="${PROFILE_SECONDS:-10}"
daemon_pid=''
gateway_pid=''
perf_pid=''
owns_segment=false

[[ "$(uname -s)" == Linux ]] || { echo 'This profiler requires Linux.' >&2; exit 2; }
for value in "$request_count" "$concurrency" "$seed" "$profile_seconds"; do
  [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) || { echo "Invalid positive integer: $value" >&2; exit 2; }
done
[[ "$arrival_rate" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid rate: $arrival_rate" >&2; exit 2; }
for executable in "$daemon_bin" "$gateway_bin" "$load_bin"; do
  [[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 2; }
done

mkdir -p "$artifact_dir/load"
daemon_log="$artifact_dir/daemon.log"
gateway_log="$artifact_dir/gateway.log"
perf_diagnostics="$artifact_dir/perf-diagnostics.txt"
provenance="$artifact_dir/provenance.txt"

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

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$perf_pid" ]] && kill -0 "$perf_pid" 2>/dev/null; then
    kill -INT "$perf_pid" 2>/dev/null || true
    wait "$perf_pid" 2>/dev/null || true
  fi
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    kill -INT "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
  if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
    kill -TERM "$daemon_pid" 2>/dev/null || true
    for _ in {1..100}; do kill -0 "$daemon_pid" 2>/dev/null || break; sleep 0.02; done
    if kill -0 "$daemon_pid" 2>/dev/null; then kill -KILL "$daemon_pid" 2>/dev/null || true; fi
    wait "$daemon_pid" 2>/dev/null || true
  fi
  if [[ "$owns_segment" == true ]] && segment_exists; then
    ipcrm -M "$ipc_key" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  echo "captured_at_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=${GITHUB_SHA:-$(git -C "$repository_root" rev-parse HEAD 2>/dev/null || echo unknown)}"
  echo "kernel=$(uname -srvm)"
  echo "machine=$(uname -m)"
  echo "cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
  lscpu 2>/dev/null || true
  cmake --version | head -n 1 || true
  c++ --version | head -n 1 || true
  rustc --version || true
  cargo --version || true
  go version || true
  flatc --version || true
  perf version || true
  sysctl kernel.perf_event_paranoid kernel.kptr_restrict 2>&1 || true
} >"$provenance"

{
  echo "perf_command=perf record -F 499 -g --call-graph fp -p DAEMON_PID -- sleep $profile_seconds"
  echo "perf_event_paranoid_before=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unavailable)"
  echo "kptr_restrict=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null || echo unavailable)"
  echo 'If capture fails, this file preserves the host diagnostic; latency artifacts remain measured and usable.'
} >"$perf_diagnostics"
if command -v sudo >/dev/null 2>&1; then
  sudo -n sysctl -w kernel.perf_event_paranoid=1 >>"$perf_diagnostics" 2>&1 || \
    echo 'Unable to lower perf_event_paranoid on this runner.' >>"$perf_diagnostics"
fi
echo "perf_event_paranoid_after=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unavailable)" \
  >>"$perf_diagnostics"

if segment_exists; then echo "Refusing existing System V key $ipc_key" >&2; exit 2; fi
"$daemon_bin" "$ipc_key" 1024 4096 >"$daemon_log" 2>&1 &
daemon_pid=$!
for _ in {1..200}; do
  if segment_exists; then owns_segment=true; break; fi
  kill -0 "$daemon_pid" 2>/dev/null || break
  sleep 0.02
done
[[ "$owns_segment" == true ]] || { echo 'Daemon failed to create System V segment.' >&2; exit 1; }

FRAUD_SHM_KEY="$ipc_key" FRAUD_GATEWAY_TOKEN="$gateway_token" \
  FRAUD_GATEWAY_ADDR="$gateway_addr" FRAUD_GATEWAY_DEADLINE_MS=100 \
  "$gateway_bin" >"$gateway_log" 2>&1 &
gateway_pid=$!
base_url="http://$gateway_addr"
ready=false
for _ in {1..200}; do
  if curl --fail --silent --show-error "$base_url/health" >/dev/null; then ready=true; break; fi
  kill -0 "$gateway_pid" 2>/dev/null || break
  sleep 0.05
done
[[ "$ready" == true ]] || { echo 'Gateway failed readiness.' >&2; exit 1; }

perf_available=false
if command -v perf >/dev/null 2>&1; then
  set +e
  perf record -F 499 -g --call-graph fp -p "$daemon_pid" \
    -o "$artifact_dir/perf.data" -- sleep "$profile_seconds" \
    >>"$perf_diagnostics" 2>&1 &
  perf_pid=$!
  sleep 0.5
  if kill -0 "$perf_pid" 2>/dev/null; then perf_available=true; fi
  set -e
else
  echo 'perf executable unavailable.' >>"$perf_diagnostics"
fi

{
  echo "requests=$request_count"
  echo "rate_per_second=$arrival_rate"
  echo "concurrency=$concurrency"
  echo "seed=$seed"
  echo 'payload_mode=transaction'
  echo "url=$base_url/v1/transactions"
} >"$artifact_dir/workload.txt"

"$load_bin" -url "$base_url/v1/transactions" -token "$gateway_token" \
  -payload-mode transaction -entities 'profile-primary:8,profile-secondary:2' \
  -requests "$request_count" -rate "$arrival_rate" -concurrency "$concurrency" \
  -timeout 2s -seed "$seed" -output "$artifact_dir/load" \
  >"$artifact_dir/load-summary.json"

if [[ -n "$perf_pid" ]]; then
  set +e
  wait "$perf_pid"
  perf_status=$?
  set -e
  perf_pid=''
  echo "perf_exit_status=$perf_status" >>"$perf_diagnostics"
  if (( perf_status != 0 )); then perf_available=false; fi
fi

if [[ "$perf_available" == true && -s "$artifact_dir/perf.data" ]]; then
  if perf script -i "$artifact_dir/perf.data" >"$artifact_dir/perf.script" 2>>"$perf_diagnostics" && \
     [[ -x "$flamegraph_dir/stackcollapse-perf.pl" && -x "$flamegraph_dir/flamegraph.pl" ]]; then
    "$flamegraph_dir/stackcollapse-perf.pl" "$artifact_dir/perf.script" \
      >"$artifact_dir/perf.folded" 2>>"$perf_diagnostics"
    "$flamegraph_dir/flamegraph.pl" --title 'Fraud inference daemon: measured perf stacks' \
      "$artifact_dir/perf.folded" >"$artifact_dir/flamegraph.svg" 2>>"$perf_diagnostics"
    echo 'flamegraph_status=generated_from_perf_data' >>"$perf_diagnostics"
  else
    echo 'flamegraph_status=perf_script_or_flamegraph_tool_failed' >>"$perf_diagnostics"
  fi
else
  echo 'flamegraph_status=not_generated_perf_unavailable_or_denied' >>"$perf_diagnostics"
fi

echo "Profiling artifacts written to $artifact_dir"
