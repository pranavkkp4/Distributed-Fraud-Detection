package aggregations

import (
	"bufio"
	"context"
	"errors"
	"io"
	"math"
	"net"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRedisMaterializerExactVectorAndReuse(t *testing.T) {
	commands := make(chan []string, 2)
	address := startMaterializerRESPServer(t, func(conn net.Conn, args []string) {
		commands <- args
		_, _ = io.WriteString(conn, "+OK\r\n")
	})
	materializer, err := NewRedisMaterializer(address, 100*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer materializer.Close()
	features := Features{EntityID: "account", EventCount: 4, TotalAmount: 12.5, MaxAmount: 7}
	for range 2 {
		if err := materializer.Put(context.Background(), features); err != nil {
			t.Fatal(err)
		}
	}
	for range 2 {
		args := <-commands
		if len(args) != 3 || !strings.EqualFold(args[0], "set") || args[1] != "account" {
			t.Fatalf("unexpected Redis command: %#v", args)
		}
		vector := strings.Split(args[2], ",")
		if len(vector) != MaterializedFeatureWidth {
			t.Fatalf("feature width = %d, want %d", len(vector), MaterializedFeatureWidth)
		}
		if vector[0] != "4" || vector[1] != "12.5" || vector[2] != "7" {
			t.Fatalf("populated features = %#v", vector[:3])
		}
		for i, value := range vector[3:] {
			if value != "0" {
				t.Fatalf("reserved feature %d = %q, want 0", i+3, value)
			}
		}
	}
}

func TestRedisMaterializerProtocolDeadlineAndValidation(t *testing.T) {
	t.Run("truncated", func(t *testing.T) {
		address := startMaterializerRESPServer(t, func(conn net.Conn, _ []string) {
			_, _ = io.WriteString(conn, "+O")
			_ = conn.Close()
		})
		materializer, _ := NewRedisMaterializer(address, 30*time.Millisecond)
		defer materializer.Close()
		if err := materializer.Put(context.Background(), Features{EntityID: "a"}); err == nil {
			t.Fatal("expected truncated RESP error")
		}
	})

	t.Run("deadline", func(t *testing.T) {
		address := startMaterializerRESPServer(t, func(_ net.Conn, _ []string) {
			time.Sleep(100 * time.Millisecond)
		})
		materializer, _ := NewRedisMaterializer(address, 15*time.Millisecond)
		defer materializer.Close()
		started := time.Now()
		if err := materializer.Put(context.Background(), Features{EntityID: "a"}); err == nil {
			t.Fatal("expected deadline error")
		}
		if elapsed := time.Since(started); elapsed > 80*time.Millisecond {
			t.Fatalf("deadline took %s", elapsed)
		}
	})

	t.Run("cancellation_and_key_injection", func(t *testing.T) {
		var commands atomic.Int64
		address := startMaterializerRESPServer(t, func(conn net.Conn, _ []string) {
			commands.Add(1)
			_, _ = io.WriteString(conn, "+OK\r\n")
		})
		materializer, _ := NewRedisMaterializer(address, 30*time.Millisecond)
		defer materializer.Close()
		if err := materializer.Put(context.Background(), Features{EntityID: "a\r\nSET injected"}); err == nil {
			t.Fatal("expected key validation error")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := materializer.Put(ctx, Features{EntityID: "a"}); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled SET error = %v", err)
		}
		if commands.Load() != 0 {
			t.Fatalf("invalid requests reached Redis: %d", commands.Load())
		}
	})
}

func TestMaterializersRejectNonFiniteFeatures(t *testing.T) {
	invalid := Features{EntityID: "account", EventCount: 2, TotalAmount: math.Inf(1), MaxAmount: 1}
	memory := NewMemoryMaterializer()
	if err := memory.Put(context.Background(), invalid); err == nil {
		t.Fatal("memory materializer accepted an infinite aggregate")
	}
	if _, ok := memory.Get("account"); ok {
		t.Fatal("memory materializer stored an invalid aggregate")
	}

	var commands atomic.Int64
	address := startMaterializerRESPServer(t, func(conn net.Conn, _ []string) {
		commands.Add(1)
		_, _ = io.WriteString(conn, "+OK\r\n")
	})
	redisMaterializer, err := NewRedisMaterializer(address, 30*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer redisMaterializer.Close()
	if err := redisMaterializer.Put(context.Background(), invalid); err == nil {
		t.Fatal("Redis materializer accepted an infinite aggregate")
	}
	if commands.Load() != 0 {
		t.Fatalf("invalid aggregate reached Redis: %d commands", commands.Load())
	}
}

func startMaterializerRESPServer(t *testing.T, handler func(net.Conn, []string)) string {
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
					args, readErr := readMaterializerRESPCommand(reader)
					if readErr != nil {
						return
					}
					if len(args) > 0 && strings.EqualFold(args[0], "hello") {
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

func readMaterializerRESPCommand(reader *bufio.Reader) ([]string, error) {
	line, err := reader.ReadString('\n')
	if err != nil || !strings.HasPrefix(line, "*") {
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
		args[i] = string(buffer[:length])
	}
	return args, nil
}
