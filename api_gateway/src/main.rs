use fraud_shm_api_gateway::{app, Gateway, SharedSegment};
use std::env;
use std::net::SocketAddr;
use std::time::Duration;

fn required(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

#[tokio::main]
async fn main() {
    let key = required("FRAUD_SHM_KEY")
        .parse::<i64>()
        .expect("FRAUD_SHM_KEY must be an integer") as libc::key_t;
    let token = required("FRAUD_GATEWAY_TOKEN");
    let address: SocketAddr = env::var("FRAUD_GATEWAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8081".into())
        .parse()
        .expect("invalid FRAUD_GATEWAY_ADDR");
    let deadline_ms = env::var("FRAUD_GATEWAY_DEADLINE_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10u64);
    let gateway = Gateway::new(
        SharedSegment::attach(key).expect("attach inference shared memory"),
        token,
        Duration::from_millis(deadline_ms),
    )
    .expect("construct gateway");
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("bind gateway listener");
    axum::serve(listener, app(gateway))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("serve gateway");
}
