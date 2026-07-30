# Minikube deployment

The manifests reference local images with `imagePullPolicy: Never`. Build them
inside Minikube before applying the kustomization:

```text
minikube start --cpus=4 --memory=6144
minikube addons enable metrics-server
make images-minikube
make deploy-minikube
kubectl -n fraud-inference rollout status deployment/fraud-worker
kubectl -n fraud-inference rollout status deployment/fraud-gateway
minikube service -n fraud-inference fraud-gateway --url
```

The included Secret and anonymous Grafana settings are local-development values.
Replace `fraud-auth` through the cluster's secret manager before using a shared
environment. Redpanda and Redis are single-node, ephemeral dependencies here;
production deployments should use durable managed equivalents and appropriate
network/TLS authentication.

This kustomization is an explicit development overlay because `fraud-config`
sets `FRAUD_DEVELOPMENT_INSECURE=true`. A production overlay must remove that
flag, mount the gateway/worker/stream, Redis, and Kafka trust/identity secrets,
and change HTTP/gRPC probes to TLS-aware probes with the matching server names.
