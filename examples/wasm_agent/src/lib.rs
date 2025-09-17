#![allow(dead_code, unused_variables, unused_imports)]
use wasm_bindgen::prelude::*;

// Models available in WASM
mod agent;
pub mod phi_agent;
pub mod phi_llm_provider;
pub mod phi_provider;

pub use agent::{LLamaChatWrapper, LlamaChatAgent};
pub use phi_agent::{PhiAgentWrapper, PhiChatAgent};
pub use phi_llm_provider::PhiLLMProvider;
pub use phi_provider::PhiModel;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}

#[cfg(target_arch = "wasm32")]
pub fn init_wasm() {
    // Show panic messages & backtraces in the browser console
    console_error_panic_hook::set_once();

    // Optional: route log::info!/error!/debug! to browser console
    console_log::init_with_level(log::Level::Debug).ok();
}
