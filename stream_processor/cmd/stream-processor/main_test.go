package main

import "testing"

func TestKafkaModeRequiresRemoteMaterializer(t *testing.T) {
	t.Setenv("REDIS_ADDR", "")
	if _, _, _, err := configuredMaterializer(true); err == nil {
		t.Fatal("Kafka mode accepted a process-local materializer")
	}
	if sink, _, _, err := configuredMaterializer(false); err != nil || sink == nil {
		t.Fatalf("local stdin materializer = %T, %v", sink, err)
	}
}

func TestConfiguredConsumerWiresBuiltInDLQTopic(t *testing.T) {
	t.Setenv("KAFKA_BROKERS", "127.0.0.1:9092")
	t.Setenv("KAFKA_TOPIC", "events")
	t.Setenv("KAFKA_GROUP_ID", "tests")
	t.Setenv("KAFKA_POISON_POLICY", "dlq")
	t.Setenv("KAFKA_DLQ_TOPIC", "")
	if _, _, err := configuredConsumer(); err == nil {
		t.Fatal("DLQ policy accepted a missing topic")
	}

	t.Setenv("KAFKA_DLQ_TOPIC", "events-dlq")
	consumer, source, err := configuredConsumer()
	if err != nil {
		t.Fatal(err)
	}
	defer consumer.Close()
	if source != "kafka" {
		t.Fatalf("source = %q, want kafka", source)
	}
}
