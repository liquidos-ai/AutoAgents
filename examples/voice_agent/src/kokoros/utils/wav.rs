use std::io::{self, Write};

pub struct WavHeader {
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
}

impl WavHeader {
    pub fn new(channels: u16, sample_rate: u32, bits_per_sample: u16) -> Self {
        Self {
            channels,
            sample_rate,
            bits_per_sample,
        }
    }

    pub fn write_header<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // RIFF header
        writer.write_all(b"RIFF")?;
        writer.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?; // File size - 8 (placeholder)
        writer.write_all(b"WAVE")?;

        // Format chunk
        writer.write_all(b"fmt ")?;
        writer.write_all(&(16u32).to_le_bytes())?; // Format chunk size
        writer.write_all(&(3u16).to_le_bytes())?; // Format = 3 (IEEE float)
        writer.write_all(&self.channels.to_le_bytes())?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;
        let byte_rate =
            self.sample_rate * u32::from(self.channels) * u32::from(self.bits_per_sample) / 8;
        writer.write_all(&byte_rate.to_le_bytes())?;
        let block_align = self.channels * self.bits_per_sample / 8;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&self.bits_per_sample.to_le_bytes())?;

        // Data chunk header
        writer.write_all(b"data")?;
        writer.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?; // Data size (placeholder)

        Ok(())
    }
}

pub fn write_audio_chunk<W: Write>(writer: &mut W, samples: &[f32]) -> io::Result<()> {
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}
