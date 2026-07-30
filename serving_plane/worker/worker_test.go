package worker

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"math"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func TestCPUScoreAndGRPC(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	metadataSeen := make(chan metadata.MD, 1)
	weights := make([]float64, ModelFeatureWidth)
	weights[0], weights[1] = 1, 1
	server := ServeInsecureForDevelopment(l, CPUScorer{Weights: weights}, "secret", grpc.UnaryInterceptor(func(ctx context.Context, request any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if strings.HasSuffix(info.FullMethod, "/Score") {
			incoming, _ := metadata.FromIncomingContext(ctx)
			metadataSeen <- incoming
		}
		return handler(ctx, request)
	}))
	defer server.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	client, err := DialInsecureForDevelopment(ctx, l.Addr().String(), "secret")
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	features := make([]float64, ModelFeatureWidth)
	features[0], features[1] = 1, 2
	out, err := client.Infer(WithRequestMetadata(ctx, "request-1", ""), []Request{{Features: features}})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].Score <= 0.5 {
		t.Fatalf("bad result: %#v", out)
	}
	if out[0].Confidence < 0 || out[0].Confidence > 1 || len(out[0].Explanation) > MaxExplanations {
		t.Fatalf("unbounded result: %#v", out[0])
	}
	for _, detail := range out[0].Explanation {
		if math.Abs(detail.Impact) > 1 {
			t.Fatalf("unbounded explanation: %#v", detail)
		}
	}
	if out[0].Model == "" || out[0].CalibrationVersion == "" || out[0].PolicyVersion == "" || out[0].FeatureVersion == "" {
		t.Fatalf("missing versions: %#v", out[0])
	}
	incoming := <-metadataSeen
	requestIDs, authorizations := incoming.Get("x-request-id"), incoming.Get("authorization")
	if len(requestIDs) != 1 || len(authorizations) != 1 || requestIDs[0] != "request-1" || authorizations[0] != "Bearer secret" {
		t.Fatalf("metadata not propagated: %#v", incoming)
	}
	if err := client.Healthy(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestGRPCHealthReflectsEngine(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := ServeInsecureForDevelopment(l, &fakeInference{err: errors.New("model unavailable")}, "")
	defer server.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	client, err := DialInsecureForDevelopment(ctx, l.Addr().String(), "")
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err := client.Healthy(ctx); err == nil {
		t.Fatal("unhealthy engine reported SERVING")
	}
}

func TestProductionTransportRequiresMutualTLS(t *testing.T) {
	caFile, serverCert, serverKey, clientCert, clientKey := testPKI(t)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server, err := ServeWithTransport(
		l,
		DefaultCPUScorer(),
		"secret",
		TransportConfig{
			CAFile:            caFile,
			CertFile:          serverCert,
			KeyFile:           serverKey,
			RequireClientCert: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	client, err := DialWithTransport(ctx, l.Addr().String(), "secret", TransportConfig{
		ServerName: "localhost",
		CAFile:     caFile,
		CertFile:   clientCert,
		KeyFile:    clientKey,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err := client.Healthy(ctx); err != nil {
		t.Fatalf("mTLS health: %v", err)
	}
	features := make([]float64, ModelFeatureWidth)
	if _, err := client.Infer(ctx, []Request{{Features: features}}); err != nil {
		t.Fatalf("mTLS inference: %v", err)
	}

	missingClientCtx, missingClientCancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer missingClientCancel()
	if unauthenticated, err := DialWithTransport(
		missingClientCtx,
		l.Addr().String(),
		"secret",
		TransportConfig{ServerName: "localhost", CAFile: caFile},
	); err == nil {
		_ = unauthenticated.Close()
		t.Fatal("worker accepted a client without an mTLS identity")
	}
}

func TestObservedServerReportsListenerFailure(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server, serveErrors, err := ServeWithTransportObserved(listener, DefaultCPUScorer(), "", TransportConfig{Insecure: true})
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-serveErrors:
		if err == nil {
			t.Fatal("listener failure produced no Serve error")
		}
	case <-time.After(time.Second):
		t.Fatal("gRPC Serve failure was not reported")
	}
}

func testPKI(t *testing.T) (string, string, string, string, string) {
	t.Helper()
	directory := t.TempDir()
	now := time.Now()
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "fraud-test-ca"},
		NotBefore:             now.Add(-time.Minute),
		NotAfter:              now.Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	caFile := filepath.Join(directory, "ca.pem")
	writePEM(t, caFile, "CERTIFICATE", caDER)

	issue := func(name string, serial int64, usage x509.ExtKeyUsage, dnsNames []string) (string, string) {
		t.Helper()
		key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		template := &x509.Certificate{
			SerialNumber: big.NewInt(serial),
			Subject:      pkix.Name{CommonName: name},
			NotBefore:    now.Add(-time.Minute),
			NotAfter:     now.Add(time.Hour),
			KeyUsage:     x509.KeyUsageDigitalSignature,
			ExtKeyUsage:  []x509.ExtKeyUsage{usage},
			DNSNames:     dnsNames,
		}
		certificate, err := x509.CreateCertificate(rand.Reader, template, caTemplate, &key.PublicKey, caKey)
		if err != nil {
			t.Fatal(err)
		}
		certificateFile := filepath.Join(directory, name+".pem")
		keyFile := filepath.Join(directory, name+"-key.pem")
		writePEM(t, certificateFile, "CERTIFICATE", certificate)
		keyDER, err := x509.MarshalPKCS8PrivateKey(key)
		if err != nil {
			t.Fatal(err)
		}
		writePEM(t, keyFile, "PRIVATE KEY", keyDER)
		return certificateFile, keyFile
	}
	serverCert, serverKey := issue("localhost", 2, x509.ExtKeyUsageServerAuth, []string{"localhost"})
	clientCert, clientKey := issue("gateway", 3, x509.ExtKeyUsageClientAuth, nil)
	return caFile, serverCert, serverKey, clientCert, clientKey
}

func writePEM(t *testing.T, path, blockType string, contents []byte) {
	t.Helper()
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if err := pem.Encode(file, &pem.Block{Type: blockType, Bytes: contents}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRPCRequiresExactly32Features(t *testing.T) {
	_, err := requestsFromProto(requestsToProto([]Request{{Features: []float64{1}}}))
	if err == nil {
		t.Fatal("expected exact feature-width error")
	}
}

type canceledInference struct{ err error }

func (f canceledInference) Infer(context.Context, []Request) ([]Result, error) { return nil, f.err }
func (f canceledInference) Healthy(context.Context) error                      { return nil }

func TestGRPCCancellationStatus(t *testing.T) {
	features := make([]float64, ModelFeatureWidth)
	for expected, cause := range map[codes.Code]error{codes.Canceled: context.Canceled, codes.DeadlineExceeded: context.DeadlineExceeded} {
		_, err := (Service{Engine: canceledInference{err: cause}}).Score(context.Background(), requestsToProto([]Request{{Features: features}}))
		if status.Code(err) != expected {
			t.Fatalf("status = %v, want %v", status.Code(err), expected)
		}
	}
}

func TestInferenceBounds(t *testing.T) {
	scorer := CPUScorer{Weights: []float64{1}}
	if _, err := scorer.Infer(context.Background(), []Request{{Features: []float64{math.NaN()}}}); err == nil {
		t.Fatal("expected non-finite feature error")
	}
	requests := make([]Request, MaxBatchSize+1)
	for i := range requests {
		requests[i] = Request{Features: []float64{1}}
	}
	if _, err := scorer.Infer(context.Background(), requests); err == nil {
		t.Fatal("expected batch bound error")
	}
	if width := len(DefaultCPUScorer().Weights); width != ModelFeatureWidth {
		t.Fatalf("default feature width = %d, want %d", width, ModelFeatureWidth)
	}
}

type fakeInference struct {
	err   error
	calls atomic.Uint64
}

func (f *fakeInference) Infer(context.Context, []Request) ([]Result, error) {
	f.calls.Add(1)
	if f.err != nil {
		return nil, f.err
	}
	return []Result{{Score: 0.1}}, nil
}

func (f *fakeInference) Healthy(context.Context) error { return f.err }

func TestPoolFallsThroughFailedWorker(t *testing.T) {
	failed := &fakeInference{err: errors.New("down")}
	healthy := &fakeInference{}
	pool, err := NewPool([]Inference{failed, healthy}, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	results, err := pool.Infer(context.Background(), []Request{{Features: []float64{1}}})
	if err != nil || len(results) != 1 {
		t.Fatalf("pool result = %#v, %v", results, err)
	}
	if failed.calls.Load() != 1 || healthy.calls.Load() != 1 {
		t.Fatalf("worker calls failed=%d healthy=%d", failed.calls.Load(), healthy.calls.Load())
	}
}

type stalledInference struct{}

func (stalledInference) Infer(ctx context.Context, _ []Request) ([]Result, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (stalledInference) Healthy(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

func TestPoolPreservesBudgetForHealthyReplica(t *testing.T) {
	healthy := &fakeInference{}
	pool, err := NewPool([]Inference{stalledInference{}, healthy}, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	started := time.Now()
	results, err := pool.Infer(ctx, []Request{{Features: []float64{1}}})
	if err != nil || len(results) != 1 {
		t.Fatalf("pool inference = %#v, %v", results, err)
	}
	if elapsed := time.Since(started); elapsed >= 180*time.Millisecond {
		t.Fatalf("stalled replica consumed failover budget: %s", elapsed)
	}

	readinessPool, err := NewPool([]Inference{stalledInference{}, healthy}, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	started = time.Now()
	if err := readinessPool.Healthy(ctx); err != nil {
		t.Fatalf("healthy replica was hidden by stalled readiness probe: %v", err)
	}
	if elapsed := time.Since(started); elapsed >= 180*time.Millisecond {
		t.Fatalf("stalled readiness probe consumed full budget: %s", elapsed)
	}
}
