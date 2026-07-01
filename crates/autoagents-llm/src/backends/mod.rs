// On native targets every backend is available as before. On WASI Preview2
// (`wasm32-wasip2`) only the OpenAI Responses backend is wired up (via the
// `wasi-http` feature); the other HTTP backends remain native-only for now.
//
// The module-level cfg guards below ensure the OpenAI backend compiles on
// `wasm32-wasip2` only when the `wasi-http` feature is enabled. The crate-level
// `compile_error!`s in `lib.rs` make the unsupported combinations fail loudly.

#[cfg(all(
    feature = "openai",
    any(
        not(target_arch = "wasm32"),
        all(
            target_arch = "wasm32",
            target_os = "wasi",
            target_env = "p2",
            feature = "wasi-http"
        )
    )
))]
pub mod openai;

#[cfg(all(feature = "anthropic", not(target_arch = "wasm32")))]
pub mod anthropic;

#[cfg(all(feature = "ollama", not(target_arch = "wasm32")))]
pub mod ollama;

#[cfg(all(feature = "deepseek", not(target_arch = "wasm32")))]
pub mod deepseek;

#[cfg(all(feature = "xai", not(target_arch = "wasm32")))]
pub mod xai;

#[cfg(all(feature = "phind", not(target_arch = "wasm32")))]
pub mod phind;

#[cfg(all(feature = "google", not(target_arch = "wasm32")))]
pub mod google;

#[cfg(all(feature = "groq", not(target_arch = "wasm32")))]
pub mod groq;

#[cfg(all(feature = "azure_openai", not(target_arch = "wasm32")))]
pub mod azure_openai;

#[cfg(all(feature = "openrouter", not(target_arch = "wasm32")))]
pub mod openrouter;

#[cfg(all(feature = "minimax", not(target_arch = "wasm32")))]
pub mod minimax;
