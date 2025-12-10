/// Basic vector distance helpers.
pub trait VectorDistance {
    fn cosine_similarity(&self, other: &Self, normalize: bool) -> f32;
}

impl VectorDistance for [f32] {
    fn cosine_similarity(&self, other: &Self, normalize: bool) -> f32 {
        let dot = self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();

        if !normalize {
            return dot;
        }

        let norm_a = self.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm_b = other.iter().map(|b| b * b).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl VectorDistance for Vec<f32> {
    fn cosine_similarity(&self, other: &Self, normalize: bool) -> f32 {
        self.as_slice()
            .cosine_similarity(other.as_slice(), normalize)
    }
}
