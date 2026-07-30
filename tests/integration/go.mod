module github.com/pranavkkp4/distributed-fraud-detection/tests/integration

go 1.24

require (
	github.com/pranavkkp4/distributed-fraud-detection/serving_plane v0.0.0
	github.com/pranavkkp4/distributed-fraud-detection/stream_processor v0.0.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/klauspost/compress v1.15.9 // indirect
	github.com/pierrec/lz4/v4 v4.1.15 // indirect
	github.com/redis/go-redis/v9 v9.21.0 // indirect
	github.com/segmentio/kafka-go v0.4.47 // indirect
	github.com/xdg-go/pbkdf2 v1.0.0 // indirect
	github.com/xdg-go/scram v1.1.2 // indirect
	github.com/xdg-go/stringprep v1.0.4 // indirect
	go.uber.org/atomic v1.11.0 // indirect
	golang.org/x/text v0.15.0 // indirect
)

replace github.com/pranavkkp4/distributed-fraud-detection/serving_plane => ../../serving_plane

replace github.com/pranavkkp4/distributed-fraud-detection/stream_processor => ../../stream_processor
