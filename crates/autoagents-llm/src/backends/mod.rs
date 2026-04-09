#[cfg(all(feature = "openai", not(target_arch = "wasm32")))]
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
