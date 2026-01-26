// src/channel.rs
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::mpsc::{Receiver, Sender, channel};

#[cfg(target_arch = "wasm32")]
pub use futures::channel::mpsc::{Receiver, Sender, channel};
