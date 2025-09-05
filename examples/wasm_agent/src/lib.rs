#![allow(dead_code, unused_variables, unused_imports)]
use wasm_bindgen::prelude::*;

pub mod phi_agent;
pub mod phi_llm_provider;
pub mod phi_provider;

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
