use clap::{ArgAction, Parser, ValueEnum};

mod basic;
mod realtime;
mod util;

#[cfg(feature = "parakeet")]
mod parakeet_example;

#[derive(Debug, ValueEnum, Clone)]
enum Usecase {
    Basic,
    Realtime,
    #[cfg(feature = "parakeet")]
    Parakeet,
}

#[derive(Parser, Debug)]
#[command(version, about = "AutoAgents Speech examples", long_about = None)]
struct Args {
    #[arg(short, long, value_enum, default_value = "basic")]
    usecase: Usecase,
    #[arg(short, long, action = ArgAction::SetTrue, help = "Write output WAV files")]
    output: bool,
    #[cfg(feature = "parakeet")]
    #[arg(long, help = "Parakeet transcription mode: file, file-stream, mic, mic-stream, or all")]
    mode: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.usecase {
        Usecase::Basic => basic::run(args.output).await?,
        Usecase::Realtime => realtime::run(args.output).await?,
        #[cfg(feature = "parakeet")]
        Usecase::Parakeet => {
            let mode_str = args.mode.unwrap_or_else(|| "all".to_string());
            let mode = mode_str
                .parse::<parakeet_example::TranscriptionMode>()
                .map_err(|e| format!("Invalid mode: {}", e))?;
            parakeet_example::run(mode).await?;
        }
    }

    Ok(())
}
