mod agent;
mod cli;
mod kokoros;
mod stt;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with Unix timestamp format and environment-based log level
    tracing_subscriber::fmt()
        .with_timer(UnixTimestampFormatter)
        .init();

    let cli = Cli::new();
    cli::run(cli).await?;
    Ok(())
}
