package worker

import (
	"context"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"os"
	"sync"

	workerpb "github.com/pranavkkp4/distributed-fraud-detection/serving_plane/worker/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	grpcHealth "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type Service struct {
	workerpb.UnimplementedInferenceWorkerServer
	Engine    Inference
	AuthToken string
}

// TransportConfig deliberately makes plaintext an explicit development-only
// opt-in. Production callers must supply a server certificate and a trust
// root; ClientCA enables mutual TLS when RequireClientCert is true.
type TransportConfig struct {
	Insecure          bool
	ServerName        string
	CAFile            string
	CertFile          string
	KeyFile           string
	RequireClientCert bool
}

func (s Service) Score(ctx context.Context, request *workerpb.ScoreRequest) (*workerpb.ScoreResponse, error) {
	if s.Engine == nil {
		return nil, status.Error(codes.Unavailable, "worker unavailable")
	}
	if s.AuthToken != "" {
		values := metadata.ValueFromIncomingContext(ctx, "authorization")
		expected := []byte("Bearer " + s.AuthToken)
		if len(values) != 1 || subtle.ConstantTimeCompare([]byte(values[0]), expected) != 1 {
			return nil, status.Error(codes.Unauthenticated, "unauthorized")
		}
	}
	requests, err := requestsFromProto(request)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	results, err := s.Engine.Infer(ctx, requests)
	if err != nil {
		return nil, contextStatus(err, "inference failed")
	}
	if err := validateResults(results, len(requests)); err != nil {
		return nil, status.Error(codes.Internal, "invalid inference result")
	}
	return resultsToProto(results), nil
}

// GRPCClient holds one connection for a remote worker endpoint. Close it during shutdown.
type GRPCClient struct {
	conn      *grpc.ClientConn
	client    workerpb.InferenceWorkerClient
	authToken string
	closeOnce sync.Once
}

type requestMetadataKey struct{}
type requestMetadata struct {
	requestID     string
	authorization string
}

// WithRequestMetadata preserves request identity and caller authentication across
// the gateway-to-worker hop. A configured client token takes precedence.
func WithRequestMetadata(ctx context.Context, requestID, authorization string) context.Context {
	return context.WithValue(ctx, requestMetadataKey{}, requestMetadata{requestID: requestID, authorization: authorization})
}

func DialInsecureForDevelopment(ctx context.Context, address, authToken string) (*GRPCClient, error) {
	// The deliberately verbose name keeps plaintext out of production call sites
	// by accident. Deployment entry points use DialWithTransport.
	return DialWithTransport(ctx, address, authToken, TransportConfig{Insecure: true})
}

func DialWithTransport(ctx context.Context, address, authToken string, config TransportConfig) (*GRPCClient, error) {
	if address == "" {
		return nil, errors.New("worker address is required")
	}
	transport, err := clientCredentials(config)
	if err != nil {
		return nil, err
	}
	conn, err := grpc.DialContext(
		ctx,
		address,
		grpc.WithTransportCredentials(transport),
		grpc.WithBlock(),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(MaxRPCMessageBytes)),
	)
	if err != nil {
		return nil, err
	}
	return &GRPCClient{conn: conn, client: workerpb.NewInferenceWorkerClient(conn), authToken: authToken}, nil
}

func clientCredentials(config TransportConfig) (credentials.TransportCredentials, error) {
	if config.Insecure {
		return insecure.NewCredentials(), nil
	}
	if config.CAFile == "" {
		return nil, errors.New("worker TLS CA file is required unless development insecure mode is enabled")
	}
	pem, err := os.ReadFile(config.CAFile)
	if err != nil {
		return nil, fmt.Errorf("read worker TLS CA: %w", err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(pem) {
		return nil, errors.New("worker TLS CA contains no certificates")
	}
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS13, RootCAs: roots, ServerName: config.ServerName}
	if (config.CertFile == "") != (config.KeyFile == "") {
		return nil, errors.New("worker client TLS cert and key must be supplied together")
	}
	if config.CertFile != "" {
		cert, err := tls.LoadX509KeyPair(config.CertFile, config.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("load worker client certificate: %w", err)
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}
	return credentials.NewTLS(tlsConfig), nil
}

func ServerCredentials(config TransportConfig) (credentials.TransportCredentials, error) {
	if config.Insecure {
		return nil, nil
	}
	if config.CertFile == "" || config.KeyFile == "" {
		return nil, errors.New("worker TLS certificate and key are required unless development insecure mode is enabled")
	}
	cert, err := tls.LoadX509KeyPair(config.CertFile, config.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("load worker TLS certificate: %w", err)
	}
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{cert}}
	if config.RequireClientCert {
		if config.CAFile == "" {
			return nil, errors.New("worker client CA is required for mutual TLS")
		}
		pem, err := os.ReadFile(config.CAFile)
		if err != nil {
			return nil, fmt.Errorf("read worker client CA: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, errors.New("worker client CA contains no certificates")
		}
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		tlsConfig.ClientCAs = pool
	}
	return credentials.NewTLS(tlsConfig), nil
}

func (c *GRPCClient) Infer(ctx context.Context, requests []Request) ([]Result, error) {
	if c == nil || c.conn == nil || c.client == nil {
		return nil, errors.New("worker client is closed")
	}
	if err := validateRequests(requests, false); err != nil {
		return nil, err
	}
	if value, ok := ctx.Value(requestMetadataKey{}).(requestMetadata); ok {
		if value.requestID != "" {
			ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", value.requestID)
		}
		if c.authToken == "" && value.authorization != "" {
			ctx = metadata.AppendToOutgoingContext(ctx, "authorization", value.authorization)
		}
	}
	if c.authToken != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+c.authToken)
	}
	response, err := c.client.Score(ctx, requestsToProto(requests))
	if err != nil {
		return nil, err
	}
	results := resultsFromProto(response)
	if err := validateResults(results, len(requests)); err != nil {
		return nil, err
	}
	return results, nil
}

func (c *GRPCClient) Healthy(ctx context.Context) error {
	if c == nil || c.conn == nil {
		return errors.New("worker client is closed")
	}
	response, err := grpcHealth.NewHealthClient(c.conn).Check(ctx, &grpcHealth.HealthCheckRequest{})
	if err != nil {
		return err
	}
	if response.GetStatus() != grpcHealth.HealthCheckResponse_SERVING {
		return fmt.Errorf("worker health status is %s", response.GetStatus())
	}
	return nil
}

func (c *GRPCClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	var err error
	c.closeOnce.Do(func() { err = c.conn.Close() })
	return err
}

type engineHealthServer struct {
	grpcHealth.UnimplementedHealthServer
	engine Inference
}

func (s engineHealthServer) Check(ctx context.Context, _ *grpcHealth.HealthCheckRequest) (*grpcHealth.HealthCheckResponse, error) {
	servingStatus := grpcHealth.HealthCheckResponse_NOT_SERVING
	if s.engine != nil && s.engine.Healthy(ctx) == nil {
		servingStatus = grpcHealth.HealthCheckResponse_SERVING
	}
	return &grpcHealth.HealthCheckResponse{Status: servingStatus}, nil
}

func serve(listener net.Listener, engine Inference, authToken string, options ...grpc.ServerOption) (*grpc.Server, <-chan error) {
	options = append([]grpc.ServerOption{grpc.MaxRecvMsgSize(MaxRPCMessageBytes)}, options...)
	server := grpc.NewServer(options...)
	workerpb.RegisterInferenceWorkerServer(server, Service{Engine: engine, AuthToken: authToken})
	grpcHealth.RegisterHealthServer(server, engineHealthServer{engine: engine})
	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- server.Serve(listener)
		close(serveErrors)
	}()
	return server, serveErrors
}

func ServeInsecureForDevelopment(listener net.Listener, engine Inference, authToken string, options ...grpc.ServerOption) *grpc.Server {
	server, _ := serve(listener, engine, authToken, options...)
	return server
}

func ServeWithTransport(listener net.Listener, engine Inference, authToken string, config TransportConfig, options ...grpc.ServerOption) (*grpc.Server, error) {
	server, _, err := ServeWithTransportObserved(listener, engine, authToken, config, options...)
	return server, err
}

// ServeWithTransportObserved exposes the terminal Serve result so production
// entry points can fail readiness and exit if the gRPC accept loop dies.
func ServeWithTransportObserved(listener net.Listener, engine Inference, authToken string, config TransportConfig, options ...grpc.ServerOption) (*grpc.Server, <-chan error, error) {
	transport, err := ServerCredentials(config)
	if err != nil {
		return nil, nil, err
	}
	if transport != nil {
		options = append(options, grpc.Creds(transport))
	}
	server, serveErrors := serve(listener, engine, authToken, options...)
	return server, serveErrors, nil
}

func requestsFromProto(request *workerpb.ScoreRequest) ([]Request, error) {
	if request == nil {
		return nil, errors.New("request is required")
	}
	requests := make([]Request, len(request.Requests))
	for i, item := range request.Requests {
		if item == nil {
			return nil, errors.New("batch contains an empty request")
		}
		requests[i] = Request{Features: append([]float64(nil), item.Features...), TransactionAmount: item.TransactionAmount}
	}
	if err := validateRPCRequests(requests); err != nil {
		return nil, err
	}
	return requests, nil
}

func validateRPCRequests(requests []Request) error {
	if err := validateRequests(requests, false); err != nil {
		return err
	}
	for i, request := range requests {
		if len(request.Features) != ModelFeatureWidth {
			return fmt.Errorf("request %d feature width must be exactly %d", i, ModelFeatureWidth)
		}
	}
	return nil
}

func contextStatus(err error, fallback string) error {
	if errors.Is(err, context.Canceled) {
		return status.Error(codes.Canceled, "inference canceled")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return status.Error(codes.DeadlineExceeded, "inference deadline exceeded")
	}
	return status.Error(codes.Internal, fallback)
}

func requestsToProto(requests []Request) *workerpb.ScoreRequest {
	result := &workerpb.ScoreRequest{Requests: make([]*workerpb.FeatureRequest, len(requests))}
	for i, item := range requests {
		result.Requests[i] = &workerpb.FeatureRequest{Features: append([]float64(nil), item.Features...), TransactionAmount: item.TransactionAmount}
	}
	return result
}

func resultsToProto(results []Result) *workerpb.ScoreResponse {
	response := &workerpb.ScoreResponse{Results: make([]*workerpb.Score, len(results))}
	for i, result := range results {
		explanation := make([]*workerpb.Explanation, len(result.Explanation))
		for j, item := range result.Explanation {
			explanation[j] = &workerpb.Explanation{FeatureIndex: uint32(item.FeatureIndex), Impact: item.Impact}
		}
		response.Results[i] = &workerpb.Score{
			Score:              result.Score,
			Confidence:         result.Confidence,
			Explanation:        explanation,
			Model:              result.Model,
			CalibrationVersion: result.CalibrationVersion,
			PolicyVersion:      result.PolicyVersion,
			FeatureVersion:     result.FeatureVersion,
			Fallback:           result.Fallback,
			FallbackReason:     result.FallbackReason,
		}
	}
	return response
}

func resultsFromProto(response *workerpb.ScoreResponse) []Result {
	if response == nil {
		return nil
	}
	results := make([]Result, len(response.Results))
	for i, item := range response.Results {
		if item == nil {
			continue
		}
		explanation := make([]Explanation, 0, len(item.Explanation))
		for _, detail := range item.Explanation {
			if detail != nil && len(explanation) < MaxExplanations {
				explanation = append(explanation, Explanation{FeatureIndex: int(detail.FeatureIndex), Impact: detail.Impact})
			}
		}
		results[i] = Result{
			Score:              item.Score,
			Confidence:         item.Confidence,
			Explanation:        explanation,
			Model:              item.Model,
			CalibrationVersion: item.CalibrationVersion,
			PolicyVersion:      item.PolicyVersion,
			FeatureVersion:     item.FeatureVersion,
			Fallback:           item.Fallback,
			FallbackReason:     item.FallbackReason,
		}
	}
	return results
}
