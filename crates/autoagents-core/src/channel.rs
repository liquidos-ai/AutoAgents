// src/channel.rs
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::mpsc::{channel, Receiver, Sender};

#[cfg(target_arch = "wasm32")]
pub use futures::channel::mpsc::{channel, Receiver, Sender};
