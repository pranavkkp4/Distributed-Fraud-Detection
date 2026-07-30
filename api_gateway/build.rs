use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn normalized_generated_source(path: &Path) -> Vec<u8> {
    fs::read(path)
        .expect("read generated binding")
        .split(|byte| *byte == b'\r')
        .flat_map(|chunk| chunk.iter().copied())
        .collect()
}

fn main() {
    let schema = PathBuf::from("../schemas/transaction.fbs");
    println!("cargo:rerun-if-changed={}", schema.display());

    // The checked-in Rust binding is produced by flatc and is used directly by
    // the gateway. When flatc is available, regenerate it in OUT_DIR and fail
    // the build if the checked-in binding has gone stale. flatc is optional for
    // developer tests; release CI should set FRAUD_REQUIRE_FLATC=1.
    match which::which("flatc") {
        Ok(flatc) => {
            let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
            let status = Command::new(flatc)
                .args(["--rust", "--cpp", "-o"])
                .arg(&out)
                .arg(&schema)
                .status()
                .expect("run flatc");
            assert!(status.success(), "flatc rejected transaction.fbs");
            let generated = out.join("transaction_generated.rs");
            let checked_in = PathBuf::from("src/generated/transaction_generated.rs");
            assert_eq!(
                normalized_generated_source(&generated),
                normalized_generated_source(&checked_in),
                "generated Rust binding is stale; run flatc --rust --cpp -o api_gateway/src/generated schemas/transaction.fbs"
            );
        }
        Err(_) if env::var_os("FRAUD_REQUIRE_FLATC").is_some() => {
            panic!("FRAUD_REQUIRE_FLATC=1 but flatc was not found on PATH");
        }
        Err(_) => println!("cargo:warning=flatc absent; schema generation skipped"),
    }
}
