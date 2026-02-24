use autoagents_speech::playback::AudioPlayer;
use autoagents_speech::providers::pocket_tts::PocketTTS;
use autoagents_speech::{AudioFormat, SpeechRequest, TTSSpeechProvider, VoiceIdentifier};
use std::path::PathBuf;

#[derive(Debug)]
pub struct TtsArgs {
    pub text: String,
    pub output_file: Option<PathBuf>,
    pub voice: Option<String>,
}

pub async fn run(args: TtsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let provider = PocketTTS::new(None)?;

    let request = SpeechRequest {
        text: args.text,
        voice: VoiceIdentifier::new(args.voice.as_deref().unwrap_or("alba")),
        format: AudioFormat::Wav,
        sample_rate: Some(24_000),
    };

    let response = provider.generate_speech(request).await?;
    println!(
        "Generated {} samples at {} Hz ({} ms)",
        response.audio.samples.len(),
        response.audio.sample_rate,
        response.duration_ms
    );

    if let Some(output) = args.output_file {
        save_audio_to_file(&response.audio.samples, response.audio.sample_rate, &output)?;
        println!("Saved audio to {}", output.display());
    }

    if let Ok(player) = AudioPlayer::try_new() {
        player.play_samples(&response.audio.samples, response.audio.sample_rate);
        player.wait_until_end();
    }

    Ok(())
}

fn save_audio_to_file(
    samples: &[f32],
    sample_rate: u32,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}
