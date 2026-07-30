#!/usr/bin/env python3
"""Dependency-free gateway smoke and small-load harness.

It reports SKIP (exit 0) when the gateway cannot be reached. This is deliberate:
the harness is useful in a source checkout before local services have started.
It is a correctness smoke tool, not an SLO or benchmark certification tool.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request


def request(url: str, method: str, path: str, token: str | None = None, body: dict | None = None) -> tuple[int, dict, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            payload = response.read().decode("utf-8")
            return response.status, dict(response.headers), json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8")
        try:
            decoded = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            decoded = {"raw": payload}
        return error.code, dict(error.headers), decoded


def score_once(url: str, token: str) -> tuple[int, float]:
    started = time.perf_counter()
    status, _, payload = request(url, "POST", "/v1/score", token, {"entity_id": "smoke-missing-feature", "amount": 1.0})
    elapsed_ms = (time.perf_counter() - started) * 1000
    if status not in (200, 503):
        raise RuntimeError("unexpected score status %s: %s" % (status, payload))
    if not isinstance(payload.get("score"), (int, float)) or not isinstance(payload.get("fallback"), bool):
        raise RuntimeError("score response did not meet minimum schema: %s" % payload)
    return status, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Gateway smoke/load harness; not a performance certification.")
    parser.add_argument("--url", default=os.getenv("DFD_BASE_URL", "http://127.0.0.1:8080"), help="gateway base URL")
    parser.add_argument("--token", default=os.getenv("DFD_AUTH_TOKEN"), help="bearer token (or DFD_AUTH_TOKEN)")
    parser.add_argument("--requests", type=int, default=1, help="authenticated score requests to issue")
    parser.add_argument("--concurrency", type=int, default=1, help="maximum concurrent score requests")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")

    try:
        status, _, payload = request(url, "GET", "/healthz")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print("SKIP: gateway unavailable at %s (%s)" % (url, error))
        return 0
    if status != 200 or payload.get("status") != "ok":
        print("FAIL: health contract returned %s: %s" % (status, payload), file=sys.stderr)
        return 1
    print("PASS: health endpoint is reachable")

    if not args.token:
        print("SKIP: score/auth checks require --token or DFD_AUTH_TOKEN")
        return 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda _: score_once(url, args.token), range(args.requests)))
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as error:
        print("FAIL: score smoke check failed: %s" % error, file=sys.stderr)
        return 1
    latencies = [result[1] for result in results]
    statuses = {status: sum(1 for got, _ in results if got == status) for status in sorted({got for got, _ in results})}
    print("PASS: %d score requests completed; statuses=%s; latency_ms min=%.2f max=%.2f" % (len(results), statuses, min(latencies), max(latencies)))
    print("NOTE: results are smoke observations, not a throughput or latency-SLO pass claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
