GO ?= go
PYTHON ?= python
CMAKE ?= cmake
CTEST ?= ctest
CARGO ?= cargo
FLATC ?= flatc
MAVEN ?= mvn
DOCKER ?= docker
KUBECTL ?= kubectl
BUILD_ROOT ?= build
BUILD_CONFIG ?= Release
CPU_BUILD := $(BUILD_ROOT)/execution-cpu
CUDA_BUILD := $(BUILD_ROOT)/execution-cuda
DAEMON_BUILD := $(BUILD_ROOT)/inference-daemon

.DEFAULT_GOAL := help

.PHONY: help fmt build-cpu build-cuda build-daemon build-rust test test-python test-go test-go-race \
	test-cpp test-daemon test-rust test-load test-ipc verify-flatbuffers test-integration test-smoke test-api validate-config compose-config compose-up \
	compose-down images-minikube deploy-minikube clean

help:
	@$(PYTHON) -c "import re; p=open('Makefile', encoding='utf-8').read().splitlines(); print('Fraud inference engine targets:'); [print('  '+m.group(1)) for line in p if (m:=re.match(r'^([a-z][a-z0-9-]+):', line))]"

fmt:
	cd serving_plane && $(GO) fmt ./...
	cd stream_processor && $(GO) fmt ./...
	cd tests/integration && $(GO) fmt ./...
	cd load_tester && $(GO) fmt ./...
	cd api_gateway && $(CARGO) fmt --all

build-cpu:
	$(CMAKE) -S execution_engine -B $(CPU_BUILD) \
		-DFRAUD_ENGINE_ENABLE_CUDA=OFF -DFRAUD_ENGINE_BUILD_TESTS=ON
	$(CMAKE) --build $(CPU_BUILD) --config $(BUILD_CONFIG) --parallel

build-cuda:
	$(CMAKE) -S execution_engine -B $(CUDA_BUILD) \
		-DFRAUD_ENGINE_ENABLE_CUDA=ON -DFRAUD_ENGINE_BUILD_TESTS=ON
	$(CMAKE) --build $(CUDA_BUILD) --config $(BUILD_CONFIG) --parallel

build-daemon:
	$(CMAKE) -S inference_daemon -B $(DAEMON_BUILD) \
		-DFRAUD_DAEMON_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
	$(CMAKE) --build $(DAEMON_BUILD) --config $(BUILD_CONFIG) --parallel

build-rust:
	cd api_gateway && $(CARGO) build --release --locked

test: test-python test-go test-cpp test-daemon test-rust test-load test-integration validate-config

test-python:
	$(PYTHON) -m unittest tests/training/test_training_plane.py -v

test-go:
	cd serving_plane && $(GO) test ./...
	cd stream_processor && $(GO) test ./...

test-go-race:
	cd serving_plane && $(GO) test -race ./...
	cd stream_processor && $(GO) test -race ./...
	cd tests/integration && $(GO) test -race ./...

test-cpp: build-cpu
	$(CTEST) --test-dir $(CPU_BUILD) -C $(BUILD_CONFIG) --output-on-failure

test-daemon: build-daemon
	$(CTEST) --test-dir $(DAEMON_BUILD) -C $(BUILD_CONFIG) --output-on-failure

test-rust:
	cd api_gateway && $(CARGO) fmt --all -- --check
	cd api_gateway && $(CARGO) clippy --locked --all-targets -- -D warnings
	cd api_gateway && $(CARGO) test --locked --all-targets

test-load:
	cd load_tester && test -z "$$(gofmt -l .)"
	cd load_tester && $(GO) vet ./...
	cd load_tester && $(GO) test ./...

verify-flatbuffers:
	bash scripts/verify_flatbuffers.sh $(FLATC)

test-ipc: build-daemon build-rust
	FRAUD_DAEMON_BIN="$(abspath $(DAEMON_BUILD))/fraud_inference_daemon" \
	FRAUD_GATEWAY_BIN="$(abspath api_gateway/target/release/fraud-shm-api-gateway)" \
		bash scripts/test_ipc_integration.sh

test-integration:
	cd tests/integration && $(GO) test ./...

test-smoke:
	$(PYTHON) tests/integration/smoke_load.py

test-api:
	$(MAVEN) -q -f tests/api/pom.xml test

validate-config:
	$(PYTHON) scripts/validate_configs.py

compose-config:
	$(DOCKER) compose -f infrastructure/compose/docker-compose.yml config --quiet

compose-up:
	$(DOCKER) compose -f infrastructure/compose/docker-compose.yml up --build -d

compose-down:
	$(DOCKER) compose -f infrastructure/compose/docker-compose.yml down

images-minikube:
	minikube image build -t fraud-inference/gateway:dev -f infrastructure/docker/Dockerfile.gateway .
	minikube image build -t fraud-inference/worker:dev -f infrastructure/docker/Dockerfile.worker .
	minikube image build -t fraud-inference/stream-processor:dev -f infrastructure/docker/Dockerfile.stream-processor .

deploy-minikube:
	$(KUBECTL) apply -k infrastructure/minikube

clean:
	$(CMAKE) -E remove_directory $(CPU_BUILD)
	$(CMAKE) -E remove_directory $(CUDA_BUILD)
	$(CMAKE) -E remove_directory $(DAEMON_BUILD)
