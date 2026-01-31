//! Type conversions between pocket-tts and autoagents-tts types

use crate::error::{PocketTTSError, Result};
use autoagents_tts::{AudioData, SpeechResponse, VoiceStateData};

/// Convert pocket-tts Tensor audio to autoagents AudioData
///
/// pocket-tts generates audio as Candle Tensor in various shapes:
/// - [channels, samples] for full generation
/// - [batch, channels, samples] for streaming chunks
/// We convert it to raw PCM f32 samples and wrap in AudioData
pub fn tensor_to_audio_data(tensor: &candle_core::Tensor, sample_rate: u32) -> Result<AudioData> {
    use candle_core::DType;

    let shape = tensor.shape();
    let dims = shape.dims();

    // Handle different tensor shapes
    let (tensor_2d, channels) = match dims.len() {
        2 => {
            // Shape: [channels, samples]
            let channels = dims[0];
            (tensor.clone(), channels)
        }
        3 => {
            // Shape: [batch, channels, samples]
            // Squeeze out batch dimension (should be 1)
            if dims[0] != 1 {
                return Err(PocketTTSError::GenerationError(format!(
                    "Expected batch size 1, got {}",
                    dims[0]
                )));
            }
            let channels = dims[1];
            let squeezed = tensor.squeeze(0).map_err(|e| {
                PocketTTSError::GenerationError(format!("Failed to squeeze tensor: {}", e))
            })?;
            (squeezed, channels)
        }
        _ => {
            return Err(PocketTTSError::GenerationError(format!(
                "Expected 2D [channels, samples] or 3D [batch, channels, samples] tensor, got shape {:?}",
                shape
            )));
        }
    };

    // Convert to f32 if needed
    let tensor_f32 = if tensor_2d.dtype() == DType::F32 {
        tensor_2d
    } else {
        tensor_2d.to_dtype(DType::F32).map_err(|e| {
            PocketTTSError::GenerationError(format!("Failed to convert dtype: {}", e))
        })?
    };

    // Flatten to 1D vec
    let data: Vec<f32> = tensor_f32
        .flatten_all()
        .map_err(|e| PocketTTSError::GenerationError(format!("Failed to flatten tensor: {}", e)))?
        .to_vec1()
        .map_err(|e| PocketTTSError::GenerationError(format!("Failed to extract data: {}", e)))?;

    Ok(AudioData {
        samples: data,
        sample_rate,
        channels,
    })
}

/// Convert ModelState to VoiceStateData for serialization
pub fn model_state_to_voice_state_data(state: &pocket_tts::ModelState) -> Result<VoiceStateData> {
    // Wrap the ModelState in an Arc and store it in the PocketTTS variant
    use std::sync::Arc;
    Ok(VoiceStateData::PocketTTS(Arc::new(state.clone())))
}

/// Convert VoiceStateData to ModelState for loading
pub fn voice_state_data_to_model_state(data: &VoiceStateData) -> Result<pocket_tts::ModelState> {
    match data {
        VoiceStateData::PocketTTS(arc) => {
            // Downcast the Arc<dyn Any> to Arc<ModelState>
            arc.downcast_ref::<pocket_tts::ModelState>()
                .ok_or_else(|| {
                    PocketTTSError::VoiceLoadError(
                        "Failed to downcast VoiceStateData to ModelState".to_string(),
                    )
                })
                .map(|state| state.clone())
        }
        VoiceStateData::Other(_) => Err(PocketTTSError::VoiceLoadError(
            "Expected PocketTTS voice state data".to_string(),
        )),
    }
}

/// Create a SpeechResponse from generated audio tensor
pub fn create_speech_response(
    text: String,
    audio_tensor: &candle_core::Tensor,
    sample_rate: u32,
) -> Result<SpeechResponse> {
    let audio_data = tensor_to_audio_data(audio_tensor, sample_rate)?;

    // Calculate duration in milliseconds
    let duration_ms = (audio_data.samples.len() as f64
        / audio_data.channels as f64
        / audio_data.sample_rate as f64
        * 1000.0) as u64;

    Ok(SpeechResponse {
        text,
        audio: audio_data,
        duration_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_tensor_to_audio_data() -> anyhow::Result<()> {
        // Create a simple 1-channel, 100-sample tensor (2D)
        let device = Device::Cpu;
        let data = vec![0.0f32; 100];
        let tensor = Tensor::from_vec(data, (1, 100), &device)?;

        let audio_data = tensor_to_audio_data(&tensor, 24000)?;

        assert_eq!(audio_data.sample_rate, 24000);
        assert_eq!(audio_data.channels, 1);
        assert_eq!(audio_data.samples.len(), 100);

        Ok(())
    }

    #[test]
    fn test_tensor_3d_to_audio_data() -> anyhow::Result<()> {
        // Create a 3D tensor [batch, channels, samples] = [1, 1, 1920]
        // This is the format returned by streaming generation
        let device = Device::Cpu;
        let data = vec![0.5f32; 1920];
        let tensor = Tensor::from_vec(data, (1, 1, 1920), &device)?;

        let audio_data = tensor_to_audio_data(&tensor, 24000)?;

        assert_eq!(audio_data.sample_rate, 24000);
        assert_eq!(audio_data.channels, 1);
        assert_eq!(audio_data.samples.len(), 1920);
        // Verify the data was preserved
        assert_eq!(audio_data.samples[0], 0.5);
        assert_eq!(audio_data.samples[1919], 0.5);

        Ok(())
    }

    #[test]
    fn test_tensor_wrong_shape() {
        let device = Device::Cpu;
        let data = vec![0.0f32; 100];
        let tensor = Tensor::from_vec(data, (100,), &device).unwrap();

        let result = tensor_to_audio_data(&tensor, 24000);
        assert!(result.is_err());
    }
}
