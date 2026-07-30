// Package featurestore contains fixed-width, single-shot feature retrieval.
package featurestore

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
	"unicode"
	"unicode/utf8"

	redis "github.com/redis/go-redis/v9"
)

var ErrNotFound = errors.New("features not found")

type Store interface {
	Get(context.Context, string) ([]float64, error)
}

// Prober is optional so in-memory stores remain lightweight. Gateway readiness
// checks it when available instead of treating a configured remote store as
// healthy merely because its client object was constructed.
type Prober interface {
	Probe(context.Context) error
}

// Memory is a concurrency-safe, fixed-width fallback store.
type Memory struct {
	width  int
	mu     sync.RWMutex
	values map[string][]float64
}

func NewMemory(width int) *Memory {
	return &Memory{width: width, values: make(map[string][]float64)}
}

func (m *Memory) Put(key string, features []float64) error {
	if !validFeatureKey(key) {
		return errors.New("invalid feature key")
	}
	if len(features) != m.width {
		return fmt.Errorf("feature width must be %d", m.width)
	}
	for _, feature := range features {
		if math.IsNaN(feature) || math.IsInf(feature, 0) {
			return errors.New("features must be finite")
		}
	}
	value := append([]float64(nil), features...)
	m.mu.Lock()
	m.values[key] = value
	m.mu.Unlock()
	return nil
}

func (m *Memory) Get(ctx context.Context, key string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	value, ok := m.values[key]
	m.mu.RUnlock()
	if !ok {
		return nil, ErrNotFound
	}
	return append([]float64(nil), value...), nil
}

func (m *Memory) Probe(ctx context.Context) error { return ctx.Err() }

// Redis owns a concurrency-safe go-redis connection pool. Each Get issues one
// Redis GET and validates the complete fixed-width vector.
type Redis struct {
	client  *redis.Client
	width   int
	timeout time.Duration
}

// RedisConfig configures a feature-store connection. UseTLS enables TLS 1.3
// with the platform trust roots by default; CAFile selects a private trust root.
// Client certificates support Redis deployments that require mutual TLS.
type RedisConfig struct {
	Address    string
	Width      int
	Timeout    time.Duration
	Username   string
	Password   string
	UseTLS     bool
	CAFile     string
	ServerName string
	CertFile   string
	KeyFile    string
}

// NewRedis is the legacy local-development convenience constructor. Production
// code should use NewRedisWithConfig to explicitly select authenticated TLS.
func NewRedis(address string, width int, timeout time.Duration) (*Redis, error) {
	return NewRedisWithConfig(RedisConfig{Address: address, Width: width, Timeout: timeout})
}

func NewRedisWithConfig(config RedisConfig) (*Redis, error) {
	if config.Address == "" || config.Width < 1 {
		return nil, errors.New("Redis address and positive feature width are required")
	}
	if config.Timeout <= 0 {
		config.Timeout = 50 * time.Millisecond
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
		PoolSize:        32,
		MinIdleConns:    1,
		MaxRetries:      -1,
		DisableIdentity: true,
	})
	return &Redis{client: client, width: config.Width, timeout: config.Timeout}, nil
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

func (r *Redis) Get(ctx context.Context, key string) ([]float64, error) {
	if r == nil || r.client == nil || !validFeatureKey(key) {
		return nil, errors.New("invalid Redis feature store request")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	requestCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	value, err := r.client.Get(requestCtx, key).Result()
	if errors.Is(err, redis.Nil) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	parts := strings.Split(value, ",")
	if len(parts) != r.width {
		return nil, fmt.Errorf("Redis feature width is %d, expected %d", len(parts), r.width)
	}
	features := make([]float64, r.width)
	for i, part := range parts {
		features[i], err = strconv.ParseFloat(part, 64)
		if err != nil || math.IsNaN(features[i]) || math.IsInf(features[i], 0) {
			if err == nil {
				err = errors.New("non-finite value")
			}
			return nil, fmt.Errorf("feature %d: %w", i, err)
		}
	}
	return features, nil
}

func (r *Redis) Probe(ctx context.Context) error {
	if r == nil || r.client == nil {
		return errors.New("Redis feature store is closed")
	}
	requestCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return r.client.Ping(requestCtx).Err()
}

func (r *Redis) Close() error {
	if r == nil || r.client == nil {
		return nil
	}
	return r.client.Close()
}

// Fallback tries Primary once, then serves a local snapshot when available.
type Fallback struct {
	Primary Store
	Local   Store
}

func (f Fallback) Get(ctx context.Context, key string) ([]float64, error) {
	if f.Primary != nil {
		primaryCtx, cancel := reserveFallbackBudget(ctx)
		value, err := f.Primary.Get(primaryCtx, key)
		cancel()
		if err == nil {
			return value, nil
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if f.Local == nil {
		return nil, ErrNotFound
	}
	return f.Local.Get(ctx, key)
}

// Probe uses the primary when it is available, but permits an explicitly
// configured local fallback to keep fail-closed scoring operational during a
// transient Redis outage.
func (f Fallback) Probe(ctx context.Context) error {
	if primary, ok := f.Primary.(Prober); ok {
		primaryCtx, cancel := reserveFallbackBudget(ctx)
		err := primary.Probe(primaryCtx)
		cancel()
		if err == nil {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	if local, ok := f.Local.(Prober); ok {
		return local.Probe(ctx)
	}
	return nil
}

// reserveFallbackBudget keeps the caller's absolute deadline while giving a
// bounded primary dependency at most half of the remaining time. A timed-out
// Redis call therefore cannot consume the local fallback's entire opportunity.
func reserveFallbackBudget(ctx context.Context) (context.Context, context.CancelFunc) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return context.WithCancel(ctx)
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, remaining/2)
}

func validFeatureKey(key string) bool {
	return key != "" && len(key) <= 256 && utf8.ValidString(key) &&
		strings.TrimSpace(key) == key && strings.IndexFunc(key, unicode.IsControl) < 0
}
