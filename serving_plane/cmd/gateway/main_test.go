package main

import (
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestProbeHTTPReadyUsesConfiguredTLSCA(t *testing.T) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/readyz" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	server.StartTLS()
	defer server.Close()

	caFile := writeProbeCA(t, server.TLS.Certificates[0].Certificate[0])
	if err := probeHTTPReady(server.Listener.Addr().String(), ":8080", false, caFile, "example.com"); err != nil {
		t.Fatalf("verified HTTPS probe: %v", err)
	}
	if err := probeHTTPReady(server.Listener.Addr().String(), ":8080", false, "", "example.com"); err == nil {
		t.Fatal("HTTPS probe trusted a self-signed server without its CA")
	}
}

func TestProbeHTTPReadyUsesHTTPOnlyInDevelopment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	if err := probeHTTPReady(server.Listener.Addr().String(), ":8080", true, "", ""); err != nil {
		t.Fatalf("development HTTP probe: %v", err)
	}
}

func TestConfiguredRedisRequiresProductionAuthenticationAndTLS(t *testing.T) {
	for _, key := range []string{"REDIS_USERNAME", "REDIS_PASSWORD", "REDIS_TLS_CA_FILE", "REDIS_TLS_SERVER_NAME", "REDIS_TLS_CERT_FILE", "REDIS_TLS_KEY_FILE"} {
		t.Setenv(key, "")
	}
	if _, err := configuredRedis("redis:6379", false); err == nil {
		t.Fatal("production Redis accepted missing authentication")
	}
	t.Setenv("REDIS_USERNAME", "fraud")
	t.Setenv("REDIS_PASSWORD", "secret")
	config, err := configuredRedis("redis:6379", false)
	if err != nil {
		t.Fatal(err)
	}
	if !config.UseTLS || config.Username != "fraud" || config.Password != "secret" {
		t.Fatalf("production Redis config = %#v", config)
	}
	config, err = configuredRedis("redis:6379", true)
	if err != nil {
		t.Fatal(err)
	}
	if config.UseTLS {
		t.Fatal("development-insecure Redis unexpectedly enabled TLS")
	}
}

func writeProbeCA(t *testing.T, certificateDER []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "probe-ca.pem")
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificateDER})
	if err := os.WriteFile(path, pemBytes, 0o600); err != nil {
		t.Fatalf("write TLS CA: %v", err)
	}
	return path
}
