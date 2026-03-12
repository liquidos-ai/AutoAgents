use once_cell::sync::OnceCell;
use tokio::runtime::Runtime;

static TOKIO_RUNTIME: OnceCell<Result<Runtime, String>> = OnceCell::new();

/// Return the shared tokio runtime, initializing it on first call.
///
/// This is safe to call from any thread. The runtime is a multi-thread
/// scheduler so it can drive multiple concurrent futures.
pub fn get_runtime() -> Result<&'static Runtime, &'static str> {
    match TOKIO_RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("autoagents-py")
            .build()
            .map_err(|error| format!("tokio runtime init failed: {error}"))
    }) {
        Ok(runtime) => Ok(runtime),
        Err(message) => Err(message.as_str()),
    }
}
