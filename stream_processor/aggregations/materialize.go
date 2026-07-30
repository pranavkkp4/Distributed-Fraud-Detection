package aggregations

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// MemoryMaterializer is a concurrency-safe local feature sink.
type MemoryMaterializer struct {
	mu     sync.RWMutex
	values map[string]Features
}

func NewMemoryMaterializer() *MemoryMaterializer {
	return &MemoryMaterializer{values: map[string]Features{}}
}

func (m *MemoryMaterializer) Put(ctx context.Context, features Features) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateMaterializedFeatures(features); err != nil {
		return err
	}
	m.mu.Lock()
	m.values[features.EntityID] = features
	m.mu.Unlock()
	return nil
}

func (m *MemoryMaterializer) Get(entity string) (Features, bool) {
	m.mu.RLock()
	features, ok := m.values[entity]
	m.mu.RUnlock()
	return features, ok
}

// RedisMaterializer writes the 32-value vector consumed by the serving plane.
// The first three values are event_count,total_amount,max_amount; remaining
// values are reserved zeros. SET is idempotent, so pre-commit retries are safe.
type RedisMaterializer struct {
	client  *redis.Client
	timeout time.Duration
}

// RedisConfig configures authenticated Redis materialization. UseTLS enables
// TLS 1.3 and system trust roots; CAFile selects a private trust root. A client
// certificate/key pair enables mutual TLS where Redis requires it.
type RedisConfig struct {
	Address    string
	Timeout    time.Duration
	Username   string
	Password   string
	UseTLS     bool
	CAFile     string
	ServerName string
	CertFile   string
	KeyFile    string
}

const MaterializedFeatureWidth = 32

// NewRedisMaterializer is a legacy local-development convenience constructor.
// Production code should use NewRedisMaterializerWithConfig for explicit TLS
// and authentication settings.
func NewRedisMaterializer(address string, timeout time.Duration) (*RedisMaterializer, error) {
	return NewRedisMaterializerWithConfig(RedisConfig{Address: address, Timeout: timeout})
}

func NewRedisMaterializerWithConfig(config RedisConfig) (*RedisMaterializer, error) {
	if config.Address == "" {
		return nil, errors.New("Redis address is required")
	}
	if config.Timeout <= 0 {
		config.Timeout = 100 * time.Millisecond
	}
	tlsConfig, err := redisTLSConfig(config.UseTLS, config.CAFile, config.ServerName, config.CertFile, config.KeyFile)
	if err != nil {
		return nil, err
	}
	client := redis.NewClient(&redis.Options{
		Addr:            config.Address,
		Username:        config.Username,
		Password:        config.Password,
		TLSConfig:       tlsConfig,
		Protocol:        2,
		DialTimeout:     config.Timeout,
		ReadTimeout:     config.Timeout,
		WriteTimeout:    config.Timeout,
		PoolSize:        8,
		MinIdleConns:    1,
		MaxRetries:      -1,
		DisableIdentity: true,
	})
	return &RedisMaterializer{client: client, timeout: config.Timeout}, nil
}

func redisTLSConfig(useTLS bool, caFile, serverName, certFile, keyFile string) (*tls.Config, error) {
	if (certFile == "") != (keyFile == "") {
		return nil, errors.New("Redis TLS client certificate and key must be supplied together")
	}
	if !useTLS {
		if caFile != "" || serverName != "" || certFile != "" {
			return nil, errors.New("Redis TLS options require UseTLS=true")
		}
		return nil, nil
	}
	var roots *x509.CertPool
	if caFile != "" {
		roots = x509.NewCertPool()
		pem, err := os.ReadFile(caFile)
		if err != nil {
			return nil, fmt.Errorf("read Redis TLS CA: %w", err)
		}
		if !roots.AppendCertsFromPEM(pem) {
			return nil, errors.New("Redis TLS CA contains no certificates")
		}
	}
	config := &tls.Config{MinVersion: tls.VersionTLS13, RootCAs: roots, ServerName: serverName}
	if certFile != "" {
		certificate, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return nil, fmt.Errorf("load Redis TLS client certificate: %w", err)
		}
		config.Certificates = []tls.Certificate{certificate}
	}
	return config, nil
}

func (r *RedisMaterializer) Put(ctx context.Context, features Features) error {
	if r == nil || r.client == nil {
		return errors.New("invalid Redis materialization request")
	}
	if err := validateMaterializedFeatures(features); err != nil {
		return err
	}
	vector := make([]string, MaterializedFeatureWidth)
	for i := range vector {
		vector[i] = "0"
	}
	vector[0] = strconv.Itoa(features.EventCount)
	vector[1] = strconv.FormatFloat(features.TotalAmount, 'g', -1, 64)
	vector[2] = strconv.FormatFloat(features.MaxAmount, 'g', -1, 64)
	value := strings.Join(vector, ",")
	requestCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return r.client.Set(requestCtx, features.EntityID, value, 0).Err()
}

func validateMaterializedFeatures(features Features) error {
	if features.EntityID == "" || len(features.EntityID) > 256 || strings.ContainsAny(features.EntityID, "\r\n") ||
		features.EventCount < 0 || features.TotalAmount < 0 || features.MaxAmount < 0 ||
		math.IsNaN(features.TotalAmount) || math.IsInf(features.TotalAmount, 0) ||
		math.IsNaN(features.MaxAmount) || math.IsInf(features.MaxAmount, 0) ||
		features.MaxAmount > features.TotalAmount {
		return errors.New("invalid materialized fraud features")
	}
	return nil
}

// Probe verifies that Redis is reachable for readiness checks.
func (r *RedisMaterializer) Probe(ctx context.Context) error {
	if r == nil || r.client == nil {
		return errors.New("Redis materializer is closed")
	}
	requestCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return r.client.Ping(requestCtx).Err()
}

func (r *RedisMaterializer) Close() error {
	if r == nil || r.client == nil {
		return nil
	}
	return r.client.Close()
}
