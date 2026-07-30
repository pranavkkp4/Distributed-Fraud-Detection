#!/usr/bin/env python3
"""Parse repository-owned configuration and enforce lightweight contracts."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - developer setup failure
    raise SystemExit("PyYAML is required: python -m pip install PyYAML") from exc


ROOT = Path(__file__).resolve().parents[1]


def yaml_documents(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as source:
        return list(yaml.safe_load_all(source))


def validate() -> list[str]:
    errors: list[str] = []
    yaml_paths = sorted((ROOT / "infrastructure").rglob("*.yml")) + sorted(
        (ROOT / "infrastructure").rglob("*.yaml")
    )
    for path in yaml_paths:
        try:
            documents = yaml_documents(path)
        except yaml.YAMLError as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid YAML: {exc}")
            continue
        if not documents or any(document is None for document in documents):
            errors.append(f"{path.relative_to(ROOT)}: contains an empty YAML document")
        if "minikube" in path.parts:
            for document in documents:
                if not isinstance(document, dict):
                    errors.append(f"{path.relative_to(ROOT)}: Kubernetes document is not a mapping")
                    continue
                if not document.get("apiVersion") or not document.get("kind"):
                    errors.append(f"{path.relative_to(ROOT)}: missing apiVersion or kind")
                if document.get("kind") != "Kustomization" and not document.get("metadata", {}).get("name"):
                    errors.append(f"{path.relative_to(ROOT)}: missing metadata.name")

    compose_path = ROOT / "infrastructure/compose/docker-compose.yml"
    compose = yaml_documents(compose_path)[0]
    expected_services = {"gateway", "worker", "stream-processor", "redis", "redpanda"}
    services = compose.get("services", {})
    missing_services = expected_services - set(services)
    if missing_services:
        errors.append(f"{compose_path.relative_to(ROOT)}: missing services {sorted(missing_services)}")
    for service_name in ("gateway", "worker", "stream-processor"):
        environment = services.get(service_name, {}).get("environment", {})
        if str(environment.get("FRAUD_DEVELOPMENT_INSECURE", "")).lower() != "true":
            errors.append(
                f"{compose_path.relative_to(ROOT)}: local service {service_name!r} "
                "must make its development-insecure transport opt-in explicit"
            )
    for service_name, service in services.items():
        for published_port in service.get("ports", []):
            if isinstance(published_port, str):
                loopback_only = published_port.startswith("127.0.0.1:")
            elif isinstance(published_port, dict):
                loopback_only = published_port.get("host_ip") == "127.0.0.1"
            else:
                loopback_only = False
            if not loopback_only:
                errors.append(
                    f"{compose_path.relative_to(ROOT)}: service {service_name!r} "
                    f"publishes {published_port!r} beyond the local loopback interface"
                )

    json_paths = [
        ROOT / "docs/profiling/benchmark_manifest.example.json",
        ROOT / "infrastructure/monitoring/grafana/dashboards/fraud-overview.json",
    ]
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")

    try:
        etree.parse(ROOT / "tests/api/pom.xml")
    except (OSError, etree.ParseError) as exc:
        errors.append(f"tests/api/pom.xml: invalid XML: {exc}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("configuration validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("configuration validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
