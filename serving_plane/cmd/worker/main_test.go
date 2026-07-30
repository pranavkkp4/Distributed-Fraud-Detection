package main

import (
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestProbeReadyUsesConfiguredTLSCA(t *testing.T) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/readyz" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	server.StartTLS()
	defer server.Close()

	caFile := writeWorkerProbeCA(t, server.TLS.Certificates[0].Certificate[0])
	if err := probeReady(server.Listener.Addr().String(), false, caFile, "example.com"); err != nil {
		t.Fatalf("verified HTTPS probe: %v", err)
	}
	if err := probeReady(server.Listener.Addr().String(), false, "", "example.com"); err == nil {
		t.Fatal("HTTPS probe trusted a self-signed server without its CA")
	}
}

func TestProbeReadyUsesHTTPOnlyInDevelopment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	if err := probeReady(server.Listener.Addr().String(), true, "", ""); err != nil {
		t.Fatalf("development HTTP probe: %v", err)
	}
}

func writeWorkerProbeCA(t *testing.T, certificateDER []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "probe-ca.pem")
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificateDER})
	if err := os.WriteFile(path, pemBytes, 0o600); err != nil {
		t.Fatalf("write TLS CA: %v", err)
	}
	return path
}
