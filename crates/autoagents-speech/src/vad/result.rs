/// Voice activity status for a VAD frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadStatus {
    Speech,
    Silence,
    Unknown,
}

/// Probability thresholds for mapping VAD output to a status.
#[derive(Debug, Clone, Copy)]
pub struct VadThresholds {
    pub speech: f32,
    pub silence: f32,
}

impl VadThresholds {
    pub fn new(speech: f32, silence: f32) -> Self {
        Self { speech, silence }
    }
}

impl Default for VadThresholds {
    fn default() -> Self {
        Self {
            speech: 0.5,
            silence: 0.35,
        }
    }
}

/// Raw VAD output.
#[derive(Debug, Clone, Copy)]
pub struct VadOutput {
    pub probability: f32,
}

impl VadOutput {
    pub fn status(&self, thresholds: VadThresholds) -> VadStatus {
        if self.probability >= thresholds.speech {
            VadStatus::Speech
        } else if self.probability <= thresholds.silence {
            VadStatus::Silence
        } else {
            VadStatus::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_probabilities_to_status() {
        let thresholds = VadThresholds::default();
        let speech = VadOutput { probability: 0.9 };
        let silence = VadOutput { probability: 0.1 };
        let unknown = VadOutput { probability: 0.4 };

        assert_eq!(speech.status(thresholds), VadStatus::Speech);
        assert_eq!(silence.status(thresholds), VadStatus::Silence);
        assert_eq!(unknown.status(thresholds), VadStatus::Unknown);
    }
}
