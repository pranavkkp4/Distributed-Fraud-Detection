package featurestore

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRedisConfigRejectsIncompleteOrPlaintextTLSSettings(t *testing.T) {
	tests := []RedisConfig{
		{Address: "redis:6379", Width: 2, CertFile: "cert.pem"},
		{Address: "redis:6379", Width: 2, CAFile: "ca.pem"},
		{Address: "redis:6379", Width: 2, UseTLS: true, CAFile: "missing.pem"},
	}
	for _, config := range tests {
		if _, err := NewRedisWithConfig(config); err == nil {
			t.Fatalf("accepted invalid Redis config: %#v", config)
		}
	}
}

func TestRedisTLSCustomTrustAndAuthentication(t *testing.T) {
	certificate, caPEM := redisTestCertificate(t)
	caFile := filepath.Join(t.TempDir(), "redis-ca.pem")
	if err := os.WriteFile(caFile, caPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	var authenticated atomic.Bool
	address := startTLSRESPServer(t, certificate, func(conn net.Conn, args []string) {
		switch args[0] {
		case "hello":
			joined := strings.Join(args, " ")
			authenticated.Store(strings.Contains(joined, "auth fraud-user fraud-password"))
			_, _ = io.WriteString(conn, "%1\r\n$6\r\nserver\r\n$5\r\nredis\r\n")
		case "auth":
			if strings.Join(args, " ") == "auth fraud-user fraud-password" {
				authenticated.Store(true)
			}
			_, _ = io.WriteString(conn, "+OK\r\n")
		case "get":
			_, _ = io.WriteString(conn, "$3\r\n1,2\r\n")
		}
	})
	store, err := NewRedisWithConfig(RedisConfig{
		Address: address, Width: 2, Timeout: time.Second, Username: "fraud-user", Password: "fraud-password",
		UseTLS: true, CAFile: caFile, ServerName: "redis.test",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	features, err := store.Get(context.Background(), "account")
	if err != nil || len(features) != 2 || features[1] != 2 {
		t.Fatalf("TLS Redis GET = %#v, %v", features, err)
	}
	if !authenticated.Load() {
		t.Fatal("Redis AUTH credentials were not sent over TLS")
	}
}

func TestMemoryFixedWidth(t *testing.T) {
	memory := NewMemory(2)
	if err := memory.Put("a", []float64{1}); err == nil {
		t.Fatal("expected width error")
	}
	if err := memory.Put("a", []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	value, err := memory.Get(context.Background(), "a")
	if err != nil || value[1] != 2 {
		t.Fatal(value, err)
	}
	value[0] = 99
	again, _ := memory.Get(context.Background(), "a")
	if again[0] == 99 {
		t.Fatal("aliased values")
	}
	if err := memory.Put("non-finite", []float64{math.Inf(1), 0}); err == nil {
		t.Fatal("memory store accepted a non-finite feature")
	}
	_, err = memory.Get(context.Background(), "missing")
	if !errors.Is(err, ErrNotFound) {
		t.Fatal(err)
	}
}

type waitForCancellationStore struct{}

func (waitForCancellationStore) Get(ctx context.Context, _ string) ([]float64, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (waitForCancellationStore) Probe(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

func TestFallbackReservesCallerDeadlineForLocalSnapshot(t *testing.T) {
	local := NewMemory(2)
	if err := local.Put("account", []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	fallback := Fallback{Primary: waitForCancellationStore{}, Local: local}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	started := time.Now()
	features, err := fallback.Get(ctx, "account")
	if err != nil || len(features) != 2 || features[1] != 2 {
		t.Fatalf("fallback result = %#v, %v", features, err)
	}
	if ctx.Err() != nil || time.Since(started) >= 180*time.Millisecond {
		t.Fatalf("primary exhausted caller deadline after %s: %v", time.Since(started), ctx.Err())
	}
	probeCtx, probeCancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer probeCancel()
	if err := fallback.Probe(probeCtx); err != nil {
		t.Fatalf("fallback probe: %v", err)
	}
}

func TestRedisGETExactWidthAndReuse(t *testing.T) {
	var commands atomic.Int64
	vector := make([]string, 32)
	for i := range vector {
		vector[i] = strconv.Itoa(i)
	}
	payload := strings.Join(vector, ",")
	address := startRESPServer(t, func(conn net.Conn, args []string) {
		commands.Add(1)
		if len(args) != 2 || args[0] != "get" || args[1] != "account" {
			t.Errorf("unexpected command: %#v", args)
		}
		_, _ = fmt.Fprintf(conn, "$%d\r\n%s\r\n", len(payload), payload)
	})
	store, err := NewRedis(address, 32, 100*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for range 2 {
		features, err := store.Get(context.Background(), "account")
		if err != nil {
			t.Fatal(err)
		}
		if len(features) != 32 || features[31] != 31 {
			t.Fatalf("unexpected features: %#v", features)
		}
	}
	if commands.Load() != 2 {
		t.Fatalf("GET command count = %d, want 2", commands.Load())
	}
}

func TestRedisGETProtocolFailuresAndDeadline(t *testing.T) {
	t.Run("truncated", func(t *testing.T) {
		address := startRESPServer(t, func(conn net.Conn, _ []string) {
			_, _ = io.WriteString(conn, "$10\r\n1,2")
			_ = conn.Close()
		})
		store, _ := NewRedis(address, 3, 50*time.Millisecond)
		defer store.Close()
		if _, err := store.Get(context.Background(), "account"); err == nil {
			t.Fatal("expected truncated RESP error")
		}
	})

	t.Run("deadline", func(t *testing.T) {
		address := startRESPServer(t, func(_ net.Conn, _ []string) {
			time.Sleep(100 * time.Millisecond)
		})
		store, _ := NewRedis(address, 3, 15*time.Millisecond)
		defer store.Close()
		started := time.Now()
		if _, err := store.Get(context.Background(), "account"); err == nil {
			t.Fatal("expected deadline error")
		}
		if elapsed := time.Since(started); elapsed > 80*time.Millisecond {
			t.Fatalf("deadline took %s", elapsed)
		}
	})

	t.Run("cancellation_and_key_injection", func(t *testing.T) {
		var commands atomic.Int64
		address := startRESPServer(t, func(conn net.Conn, _ []string) {
			commands.Add(1)
			_, _ = io.WriteString(conn, "$-1\r\n")
		})
		store, _ := NewRedis(address, 3, 50*time.Millisecond)
		defer store.Close()
		if _, err := store.Get(context.Background(), "safe\r\nGET other"); err == nil {
			t.Fatal("expected key validation error")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := store.Get(ctx, "account"); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled GET error = %v", err)
		}
		if commands.Load() != 0 {
			t.Fatalf("invalid requests reached Redis: %d", commands.Load())
		}
	})
}

func startRESPServer(t *testing.T, handler func(net.Conn, []string)) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		for {
			conn, acceptErr := listener.Accept()
			if acceptErr != nil {
				return
			}
			go func() {
				defer conn.Close()
				reader := bufio.NewReader(conn)
				for {
					args, readErr := readRESPCommand(reader)
					if readErr != nil {
						return
					}
					if len(args) > 0 && args[0] == "hello" {
						_, _ = io.WriteString(conn, "%1\r\n$6\r\nserver\r\n$5\r\nredis\r\n")
						continue
					}
					handler(conn, args)
				}
			}()
		}
	}()
	return listener.Addr().String()
}

func readRESPCommand(reader *bufio.Reader) ([]string, error) {
	line, err := reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	if !strings.HasPrefix(line, "*") {
		return nil, errors.New("expected RESP array")
	}
	count, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "*")))
	if err != nil {
		return nil, err
	}
	args := make([]string, count)
	for i := range count {
		line, err = reader.ReadString('\n')
		if err != nil || !strings.HasPrefix(line, "$") {
			return nil, errors.New("expected RESP bulk string")
		}
		length, parseErr := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "$")))
		if parseErr != nil {
			return nil, parseErr
		}
		buffer := make([]byte, length+2)
		if _, err = io.ReadFull(reader, buffer); err != nil {
			return nil, err
		}
		args[i] = strings.ToLower(string(buffer[:length]))
	}
	return args, nil
}

func startTLSRESPServer(t *testing.T, certificate tls.Certificate, handler func(net.Conn, []string)) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	tlsListener := tls.NewListener(listener, &tls.Config{MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{certificate}})
	t.Cleanup(func() { _ = tlsListener.Close() })
	go func() {
		for {
			conn, acceptErr := tlsListener.Accept()
			if acceptErr != nil {
				return
			}
			go func() {
				defer conn.Close()
				reader := bufio.NewReader(conn)
				for {
					args, readErr := readRESPCommand(reader)
					if readErr != nil {
						return
					}
					handler(conn, args)
				}
			}()
		}
	}()
	return listener.Addr().String()
}

func redisTestCertificate(t *testing.T) (tls.Certificate, []byte) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "redis.test"},
		DNSNames:              []string{"redis.test"},
		NotBefore:             time.Now().Add(-time.Minute),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	certificate, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	return certificate, certPEM
}
