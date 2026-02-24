use autoagents_speech::ModelSource;
use autoagents_speech::audio_capture::{AudioCapture, AudioCaptureConfig, AudioCaptureError};
use autoagents_speech::providers::parakeet::{ModelVariant, Parakeet, ParakeetConfig};
use autoagents_speech::vad::{
    SegmentTranscription, SegmenterConfig, SileroVad, VadConfig, VadSegmenter, VadSttConfig,
    VadSttPipeline,
};
use clap::ValueEnum;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};

const VAD_HF_REPO: &str = "freddyaboulton/silero-vad";
const VAD_HF_FILE: &str = "silero_vad.onnx";
const PARAKEET_HF_REPO: &str = "altunenes/parakeet-rs";
const PARAKEET_TDT_DIR: &str = "nemotron-speech-streaming-en-0.6b";

#[derive(Debug, ValueEnum, Clone)]
pub enum InputMode {
    Mic,
    File,
}

impl FromStr for InputMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "mic" => Ok(InputMode::Mic),
            "file" => Ok(InputMode::File),
            other => Err(format!("Unsupported input mode '{other}'")),
        }
    }
}

#[derive(Debug)]
pub struct VadArgs {
    pub input: InputMode,
    pub audio_file: Option<PathBuf>,
    pub max_seconds: u64,
    pub language: Option<String>,
}

pub async fn run(args: VadArgs) -> Result<(), Box<dyn std::error::Error>> {
    let segmenter = build_vad_segmenter()?;
    let stt_provider = build_parakeet_provider()?;
    let pipeline = VadSttPipeline::new(
        segmenter,
        stt_provider,
        VadSttConfig {
            language: args.language.clone(),
            include_timestamps: false,
        },
    );

    match args.input {
        InputMode::File => run_file(&args, pipeline).await?,
        InputMode::Mic => run_mic(&args, pipeline).await?,
    }

    Ok(())
}

pub fn build_vad_segmenter() -> Result<VadSegmenter<SileroVad>, Box<dyn std::error::Error>> {
    let vad_source = ModelSource::from_hf(VAD_HF_REPO, VAD_HF_FILE);
    let vad = SileroVad::new(vad_source, VadConfig::default())?;
    let config = SegmenterConfig::default()
        .with_window_ms(30)
        .with_min_speech_ms(80)
        .with_min_silence_ms(700)
        .with_pre_roll_ms(400)
        .with_max_segment_ms(12_000)
        .with_thresholds(autoagents_speech::vad::VadThresholds::new(0.4, 0.2));
    let segmenter = VadSegmenter::new(vad, config)?;
    Ok(segmenter)
}

async fn run_file(
    args: &VadArgs,
    mut pipeline: VadSttPipeline<SileroVad, Parakeet>,
) -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = args
        .audio_file
        .clone()
        .ok_or("Provide --audio-file for file input")?;

    let capture_config = AudioCaptureConfig::default();
    let audio = AudioCapture::read_audio_with_config(&audio_path, capture_config)?;

    let mut segments = pipeline.process_audio(&audio).await?;
    if let Some(final_segment) = pipeline.finalize().await? {
        segments.push(final_segment);
    }

    if segments.is_empty() {
        println!("No speech segments detected.");
        return Ok(());
    }

    for segment in segments {
        print_segment(&segment);
    }

    Ok(())
}

async fn run_mic(
    args: &VadArgs,
    mut pipeline: VadSttPipeline<SileroVad, Parakeet>,
) -> Result<(), Box<dyn std::error::Error>> {
    let capture_config = AudioCaptureConfig::default();
    let capture = match AudioCapture::with_config(capture_config) {
        Ok(capture) => capture,
        Err(AudioCaptureError::NoInputDevice) => {
            println!("No input device available.");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let window_samples = pipeline.window_samples();
    let window_duration = Duration::from_millis(pipeline.window_ms() as u64);
    let stream = capture.start_stream()?;
    let start = Instant::now();

    println!("Listening for speech. Press Ctrl+C to stop.");

    loop {
        if args.max_seconds > 0 && start.elapsed() >= Duration::from_secs(args.max_seconds) {
            break;
        }

        tokio::time::sleep(window_duration).await;
        let chunk = match stream.read_chunk(window_samples)? {
            Some(chunk) => chunk,
            None => continue,
        };

        let segments = pipeline.process_audio(&chunk).await?;
        for segment in segments {
            print_segment(&segment);
        }
    }

    if let Some(final_segment) = pipeline.finalize().await? {
        print_segment(&final_segment);
    }

    Ok(())
}

fn print_segment(segment: &SegmentTranscription) {
    let start_sec = segment.segment.start_ms as f32 / 1000.0;
    let end_sec = segment.segment.end_ms as f32 / 1000.0;
    println!(
        "[{start_sec:.2}s - {end_sec:.2}s] {}",
        segment.transcription.text
    );
}

pub fn build_parakeet_provider() -> Result<Parakeet, Box<dyn std::error::Error>> {
    let model_path = ModelSource::from_hf_dir(PARAKEET_HF_REPO, PARAKEET_TDT_DIR).resolve()?;
    let config = ParakeetConfig::new(
        ModelVariant::Nemotron,
        model_path.to_string_lossy().to_string(),
    );
    Ok(Parakeet::new(config)?)
}
