mod agent;
mod audio;
mod cli;
mod kokoros;
mod stt;
pub(crate) mod utils;

mod ui;
mod vad;

use crate::cli::Mode;
use crate::ui::run_voice_agent_app;
use anyhow;
use cli::Cli;
use tracing_subscriber::fmt::time::FormatTime;

/// Custom Unix timestamp formatter for tracing logs
struct UnixTimestampFormatter;

impl FormatTime for UnixTimestampFormatter {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let timestamp = format!("{}.{:06}", now.as_secs(), now.subsec_micros());
        write!(w, "{}", timestamp)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with Unix timestamp format and environment-based log level
    tracing_subscriber::fmt()
        .with_timer(UnixTimestampFormatter)
        .init();

    let cli = Cli::new();
    if cli.mode != Mode::UI {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .build()
            .map_err(|_e| "Error in building runtime")?;
        runtime.block_on(async move {
            cli::run(cli).await.unwrap();
        });
    } else {
        println!("Running App");
        run_voice_agent_app().map_err(|e| anyhow::anyhow!("UI error: {}", e))?;
    }
    Ok(())
}
