use clap::Parser;
use std::path::PathBuf;

mod agent;
mod stt_example;
mod tts_example;
mod vad_stt;

#[derive(Debug, Parser)]
#[command(version, about = "AutoAgents Speech examples")]
struct Args {
    #[arg(
        long,
        value_name = "USECASE",
        help = "Which example to run: agent, vad, stt, tts"
    )]
    usecase: String,
    #[arg(long, default_value = "mic")]
    input: String,
    #[arg(long, value_name = "PATH")]
    audio_file: Option<PathBuf>,
    #[arg(long, default_value_t = 30)]
    max_seconds: u64,
    #[arg(long, value_name = "LANG")]
    language: Option<String>,
    #[arg(long)]
    text: Option<String>,
    #[arg(long, value_name = "PATH")]
    output_file: Option<PathBuf>,
    #[arg(long, value_name = "MODEL")]
    agent_model: Option<String>,
    #[arg(long, value_name = "VOICE")]
    voice: Option<String>,
}

const DEFAULT_TEXT: &str = "Hello from AutoAgents speech examples.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.usecase.as_str() {
        "agent" => {
            let input = args.input.parse().map_err(|_| {
                format!(
                    "Invalid input '{}'. Use 'mic' or 'file' for agent.",
                    args.input
                )
            })?;
            let agent_args = agent::AgentArgs {
                input,
                audio_file: args.audio_file,
                language: args.language,
                agent_model: args.agent_model,
                voice: args.voice,
            };
            agent::run(agent_args).await?;
        }
        "vad" => {
            let input = args.input.parse().map_err(|_| {
                format!(
                    "Invalid input '{}'. Use 'mic' or 'file' for VAD.",
                    args.input
                )
            })?;
            let vad_args = vad_stt::VadArgs {
                input,
                audio_file: args.audio_file,
                max_seconds: args.max_seconds,
                language: args.language,
            };
            vad_stt::run(vad_args).await?;
        }
        "stt" => {
            let audio_file = args
                .audio_file
                .ok_or("Provide --audio-file for STT example")?;
            let stt_args = stt_example::SttArgs {
                audio_file,
                language: args.language,
            };
            stt_example::run(stt_args).await?;
        }
        "tts" => {
            let text = args.text.unwrap_or_else(|| DEFAULT_TEXT.to_string());
            let tts_args = tts_example::TtsArgs {
                text,
                output_file: args.output_file,
                voice: args.voice,
            };
            tts_example::run(tts_args).await?;
        }
        other => {
            return Err(format!("Unknown usecase '{other}'. Use: agent, vad, stt, or tts.").into());
        }
    }

    Ok(())
}
