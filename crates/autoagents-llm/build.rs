//! Build-script cfg aliases for the OpenAI HTTP transports.
//!
//! To avoid repeating the long target/feature predicates throughout the crate,
//! this script derives two cfg flags:
//!
//! - `native`    — any non-`wasm32` target (reqwest/ureq, full feature set).
//! - `wasi_http` — `wasm32-wasip2` with the `wasi-http` feature
//!   (`golem-wasi-http` over the `wasi:http` interface; OpenAI Responses only).
//!
//! Browser wasm (`wasm32-unknown-unknown`) is rejected by `compile_error!` in
//! `src/lib.rs`, so it never reaches the HTTP code gated here.
//!
//! The flags are read from `CARGO_CFG_TARGET_*` rather than `#[cfg]`: a build
//! script's `#[cfg]` reflects the **host** platform, which would wrongly light
//! up `native` when cross-compiling to wasm.

/// Env-var name for the target architecture injected by Cargo.
const ENV_CARGO_CFG_TARGET_ARCH: &str = "CARGO_CFG_TARGET_ARCH";
/// Env-var name for the target OS injected by Cargo.
const ENV_CARGO_CFG_TARGET_OS: &str = "CARGO_CFG_TARGET_OS";
/// Env-var name for the target environment injected by Cargo.
const ENV_CARGO_CFG_TARGET_ENV: &str = "CARGO_CFG_TARGET_ENV";
/// Env-var name for the `wasi-http` feature flag.
const ENV_CARGO_FEATURE_WASI_HTTP: &str = "CARGO_FEATURE_WASI_HTTP";

fn main() {
    // Declare the cfg flags so a typo is a hard build error (Rust 1.80+).
    println!("cargo::rustc-check-cfg=cfg(native)");
    println!("cargo::rustc-check-cfg=cfg(wasi_http)");
    println!("cargo::rerun-if-changed=build.rs");

    // Target platform injected by cargo (not the host); see module docs.
    let target_arch = std::env::var(ENV_CARGO_CFG_TARGET_ARCH).unwrap_or_default();
    let target_os = std::env::var(ENV_CARGO_CFG_TARGET_OS).unwrap_or_default();
    let target_env = std::env::var(ENV_CARGO_CFG_TARGET_ENV).unwrap_or_default();
    let wasi_http_feature = std::env::var(ENV_CARGO_FEATURE_WASI_HTTP).is_ok();

    if target_arch != "wasm32" {
        println!("cargo::rustc-cfg=native");
    }

    // The feature is only honored on the matching target so enabling it on a
    // native build (where it is irrelevant) does not light up the wasm transport.
    if target_arch == "wasm32" && target_os == "wasi" && target_env == "p2" && wasi_http_feature {
        println!("cargo::rustc-cfg=wasi_http");
    }
}
