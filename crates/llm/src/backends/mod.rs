#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "phind")]
pub mod phind;

#[cfg(feature = "google")]
pub mod google;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "azure_openai")]
pub mod azure_openai;

#[cfg(feature = "openrouter")]
pub mod openrouter;
