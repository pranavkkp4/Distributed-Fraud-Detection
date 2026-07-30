package main

import (
	"crypto/tls"
	"net/http"
	"testing"
)

func TestKafkaModeRequiresRemoteMaterializer(t *testing.T) {
	t.Setenv("REDIS_ADDR", "")
	if _, _, _, err := configuredMaterializer(true, true); err == nil {
		t.Fatal("Kafka mode accepted a process-local materializer")
	}
	if sink, _, _, err := configuredMaterializer(false, true); err != nil || sink == nil {
		t.Fatalf("local stdin materializer = %T, %v", sink, err)
	}
}

func TestConfiguredConsumerWiresBuiltInDLQTopic(t *testing.T) {
	t.Setenv("KAFKA_BROKERS", "127.0.0.1:9092")
	t.Setenv("KAFKA_TOPIC", "events")
	t.Setenv("KAFKA_GROUP_ID", "tests")
	t.Setenv("KAFKA_POISON_POLICY", "dlq")
	t.Setenv("KAFKA_DLQ_TOPIC", "")
	if _, _, err := configuredConsumer(true); err == nil {
		t.Fatal("DLQ policy accepted a missing topic")
	}

	t.Setenv("KAFKA_DLQ_TOPIC", "events-dlq")
	consumer, source, err := configuredConsumer(true)
	if err != nil {
		t.Fatal(err)
	}
	defer consumer.Close()
	if source != "kafka" {
		t.Fatalf("source = %q, want kafka", source)
	}
}

func TestProductionDependenciesRequireAuthentication(t *testing.T) {
	for _, key := range []string{
		"KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD", "KAFKA_TLS_CERT_FILE", "KAFKA_TLS_KEY_FILE",
		"REDIS_USERNAME", "REDIS_PASSWORD", "REDIS_TLS_CERT_FILE", "REDIS_TLS_KEY_FILE",
	} {
		t.Setenv(key, "")
	}
	t.Setenv("KAFKA_BROKERS", "broker:9093")
	if _, _, err := configuredConsumer(false); err == nil {
		t.Fatal("production Kafka accepted missing authentication")
	}
	t.Setenv("KAFKA_SASL_USERNAME", "fraud-stream")
	t.Setenv("KAFKA_SASL_PASSWORD", "secret")
	config, err := configuredKafka([]string{"broker:9093"}, false)
	if err != nil || config.DevelopmentInsecure || config.SASLUsername != "fraud-stream" || config.SASLPassword != "secret" {
		t.Fatalf("authenticated production Kafka config = %#v, %v", config, err)
	}
	t.Setenv("REDIS_ADDR", "redis:6380")
	if _, _, _, err := configuredMaterializer(true, false); err == nil {
		t.Fatal("production Redis accepted missing authentication")
	}
	t.Setenv("REDIS_PASSWORD", "secret")
	redisConfig, err := configuredRedisMaterializer("redis:6380", false)
	if err != nil || !redisConfig.UseTLS || redisConfig.Password != "secret" {
		t.Fatalf("authenticated production Redis config = %#v, %v", redisConfig, err)
	}
}

func TestOperationalHTTPRequiresTLSOutsideDevelopment(t *testing.T) {
	t.Setenv("STREAM_PROCESSOR_TLS_CERT_FILE", "")
	t.Setenv("STREAM_PROCESSOR_TLS_KEY_FILE", "")
	if _, _, _, err := configuredHTTPServer(":0", http.NewServeMux(), false); err == nil {
		t.Fatal("production operational endpoint accepted plaintext")
	}
	server, _, _, err := configuredHTTPServer(":0", http.NewServeMux(), true)
	if err != nil {
		t.Fatal(err)
	}
	if server.TLSConfig == nil || server.TLSConfig.MinVersion != tls.VersionTLS13 {
		t.Fatalf("operational TLS minimum = %#v", server.TLSConfig)
	}
}
