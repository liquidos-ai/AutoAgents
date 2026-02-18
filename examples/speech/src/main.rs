use clap::{ArgAction, Parser, ValueEnum};

mod basic;
mod realtime;
mod util;
mod stt_example;

#[derive(Debug, ValueEnum, Clone)]
enum Usecase {
    Basic,
    Realtime,
    Stt,
}

#[derive(Parser, Debug)]
#[command(version, about = "AutoAgents Speech examples", long_about = None)]
struct Args {
    #[arg(short, long, value_enum, default_value = "basic")]
    usecase: Usecase,
    #[arg(short, long, action = ArgAction::SetTrue, help = "Write output WAV files")]
    output: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.usecase {
        Usecase::Basic => basic::run(args.output).await?,
        Usecase::Realtime => realtime::run(args.output).await?,
        Usecase::Stt => stt_example::run().await?,
    }

    Ok(())
}
