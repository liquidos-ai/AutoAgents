//! Utility functions and helpers for liquid-edge
//!
//! This module provides common utility functions used throughout the liquid-edge
//! crate, including file operations, validation helpers, conversion utilities,
//! and other miscellaneous functions.

use crate::error::{EdgeError, EdgeResult};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

/// File system utilities
pub mod fs {
    use super::*;
    use std::fs;

    /// Check if a path exists and is readable
    pub fn is_readable<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        path.exists() && path.is_file() && fs::metadata(path).is_ok()
    }

    /// Check if a directory exists and contains expected model files
    pub fn validate_model_directory<P: AsRef<Path>>(dir: P) -> EdgeResult<()> {
        let dir = dir.as_ref();

        if !dir.exists() {
            return Err(EdgeError::not_found(format!(
                "Model directory does not exist: {}",
                dir.display()
            )));
        }

        if !dir.is_dir() {
            return Err(EdgeError::invalid_input(format!(
                "Path is not a directory: {}",
                dir.display()
            )));
        }

        Ok(())
    }

    /// Find model files in a directory with common extensions
    pub fn find_model_files<P: AsRef<Path>>(dir: P) -> EdgeResult<Vec<PathBuf>> {
        let dir = dir.as_ref();
        validate_model_directory(dir)?;

        let mut model_files = Vec::new();
        let extensions = ["onnx", "pt", "pth", "bin", "safetensors"];

        for entry in fs::read_dir(dir).map_err(|e| EdgeError::Io { source: e })? {
            let entry = entry.map_err(|e| EdgeError::Io { source: e })?;
            let path = entry.path();

            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if extensions.contains(&ext.to_lowercase().as_str()) {
                    model_files.push(path);
                }
            }
        }

        Ok(model_files)
    }

    /// Get file size in bytes
    pub fn file_size<P: AsRef<Path>>(path: P) -> EdgeResult<u64> {
        let metadata = fs::metadata(path.as_ref()).map_err(|e| EdgeError::Io {
            source: std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to get metadata for {}: {}",
                    path.as_ref().display(),
                    e
                ),
            ),
        })?;
        Ok(metadata.len())
    }

    /// Read file to string with size limit
    pub fn read_to_string_limited<P: AsRef<Path>>(path: P, max_size: u64) -> EdgeResult<String> {
        let path = path.as_ref();
        let size = file_size(path)?;

        if size > max_size {
            return Err(EdgeError::invalid_input(format!(
                "File {} is too large: {} bytes (max: {} bytes)",
                path.display(),
                size,
                max_size
            )));
        }

        fs::read_to_string(path).map_err(|e| EdgeError::Io { source: e })
    }

    /// Create directory if it doesn't exist
    pub fn ensure_directory<P: AsRef<Path>>(path: P) -> EdgeResult<()> {
        let path = path.as_ref();
        if !path.exists() {
            fs::create_dir_all(path).map_err(|e| EdgeError::Io { source: e })?;
        }
        Ok(())
    }
}

/// String processing utilities
pub mod string {

    /// Truncate string to maximum length with ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }

    /// Clean whitespace and normalize string
    pub fn normalize_whitespace(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Remove control characters from string
    pub fn remove_control_chars(s: &str) -> String {
        s.chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect()
    }

    /// Check if string is valid UTF-8 and printable
    pub fn is_valid_text(s: &str) -> bool {
        s.chars().all(|c| !c.is_control() || c.is_whitespace())
    }

    /// Escape special characters for logging
    pub fn escape_for_log(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                c if c.is_control() => format!("\\u{:04x}", c as u32),
                c => c.to_string(),
            })
            .collect()
    }

    /// Convert bytes to human-readable string
    pub fn bytes_to_human_readable(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        const THRESHOLD: f64 = 1024.0;

        if bytes == 0 {
            return "0 B".to_string();
        }

        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
            size /= THRESHOLD;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate token ID is within vocabulary range
    pub fn validate_token_id(token_id: u32, vocab_size: usize) -> EdgeResult<()> {
        if token_id as usize >= vocab_size {
            Err(EdgeError::invalid_input(format!(
                "Token ID {} is out of vocabulary range (0-{})",
                token_id,
                vocab_size - 1
            )))
        } else {
            Ok(())
        }
    }

    /// Validate sequence length
    pub fn validate_sequence_length(length: usize, max_length: usize) -> EdgeResult<()> {
        if length > max_length {
            Err(EdgeError::invalid_input(format!(
                "Sequence length {length} exceeds maximum length {max_length}"
            )))
        } else {
            Ok(())
        }
    }

    /// Validate probability value (0.0 to 1.0)
    pub fn validate_probability(prob: f32, name: &str) -> EdgeResult<()> {
        if !(0.0..=1.0).contains(&prob) {
            Err(EdgeError::invalid_input(format!(
                "{name} must be between 0.0 and 1.0, got {prob}"
            )))
        } else {
            Ok(())
        }
    }

    /// Validate temperature value (non-negative)
    pub fn validate_temperature(temperature: f32) -> EdgeResult<()> {
        if temperature < 0.0 {
            Err(EdgeError::invalid_input(format!(
                "Temperature must be non-negative, got {temperature}"
            )))
        } else {
            Ok(())
        }
    }

    /// Validate model name (basic sanity check)
    pub fn validate_model_name(name: &str) -> EdgeResult<()> {
        if name.is_empty() {
            return Err(EdgeError::invalid_input("Model name cannot be empty"));
        }

        if name.len() > 100 {
            return Err(EdgeError::invalid_input(
                "Model name is too long (max 100 characters)",
            ));
        }

        // Check for invalid characters
        if name
            .chars()
            .any(|c| c.is_control() || "\\/:*?\"<>|".contains(c))
        {
            return Err(EdgeError::invalid_input(
                "Model name contains invalid characters",
            ));
        }

        Ok(())
    }
}

/// Timing and performance utilities
pub mod timing {
    use super::*;
    use std::fmt;

    /// Simple timer for measuring execution time
    #[derive(Debug)]
    pub struct Timer {
        start: Instant,
        name: String,
    }

    impl Timer {
        /// Create and start a new timer
        pub fn start(name: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }

        /// Get elapsed time since timer started
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        /// Get elapsed time in milliseconds
        pub fn elapsed_ms(&self) -> f64 {
            self.elapsed().as_secs_f64() * 1000.0
        }

        /// Stop timer and log the result
        pub fn stop_and_log(self) {
            log::debug!("{} took {:.2}ms", self.name, self.elapsed_ms());
        }

        /// Stop timer and return elapsed duration
        pub fn stop(self) -> Duration {
            self.elapsed()
        }
    }

    impl fmt::Display for Timer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}: {:.2}ms", self.name, self.elapsed_ms())
        }
    }

    /// Measure execution time of a function
    pub fn time_function<F, R>(name: &str, f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let timer = Timer::start(name);
        let result = f();
        let duration = timer.stop();
        (result, duration)
    }

    /// Measure and log execution time of a function
    pub fn time_and_log<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let (result, duration) = time_function(name, f);
        log::debug!("{} took {:.2}ms", name, duration.as_secs_f64() * 1000.0);
        result
    }
}

/// Conversion utilities
pub mod convert {
    use super::*;

    /// Convert slice of u32 to slice of i64
    pub fn u32_to_i64_slice(input: &[u32]) -> Vec<i64> {
        input.iter().map(|&x| x as i64).collect()
    }

    /// Convert slice of i64 to slice of u32
    pub fn i64_to_u32_slice(input: &[i64]) -> EdgeResult<Vec<u32>> {
        input
            .iter()
            .map(|&x| {
                if x < 0 || x > u32::MAX as i64 {
                    Err(EdgeError::invalid_input(format!(
                        "Value {x} cannot be converted to u32"
                    )))
                } else {
                    Ok(x as u32)
                }
            })
            .collect()
    }

    /// Convert boolean to integer (0 or 1)
    pub fn bool_to_int(b: bool) -> i64 {
        if b {
            1
        } else {
            0
        }
    }

    /// Safely convert usize to u32
    pub fn usize_to_u32(value: usize) -> EdgeResult<u32> {
        if value > u32::MAX as usize {
            Err(EdgeError::invalid_input(format!(
                "Value {value} is too large for u32"
            )))
        } else {
            Ok(value as u32)
        }
    }
}

/// Math utilities
pub mod math {
    use super::*;

    /// Calculate percentile of a sorted array
    pub fn percentile(sorted_data: &[f32], percentile: f32) -> EdgeResult<f32> {
        if sorted_data.is_empty() {
            return Err(EdgeError::invalid_input(
                "Cannot calculate percentile of empty data",
            ));
        }

        if !(0.0..=100.0).contains(&percentile) {
            return Err(EdgeError::invalid_input(
                "Percentile must be between 0 and 100",
            ));
        }

        if percentile == 0.0 {
            return Ok(sorted_data[0]);
        }
        if percentile == 100.0 {
            return Ok(sorted_data[sorted_data.len() - 1]);
        }

        let index = (percentile / 100.0) * (sorted_data.len() - 1) as f32;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Ok(sorted_data[lower])
        } else {
            let weight = index - lower as f32;
            Ok(sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight)
        }
    }

    /// Calculate mean of a slice
    pub fn mean(data: &[f32]) -> EdgeResult<f32> {
        if data.is_empty() {
            return Err(EdgeError::invalid_input(
                "Cannot calculate mean of empty data",
            ));
        }
        Ok(data.iter().sum::<f32>() / data.len() as f32)
    }

    /// Calculate standard deviation of a slice
    pub fn std_dev(data: &[f32]) -> EdgeResult<f32> {
        if data.len() < 2 {
            return Err(EdgeError::invalid_input(
                "Need at least 2 data points for standard deviation",
            ));
        }

        let mean_val = mean(data)?;
        let variance =
            data.iter().map(|x| (x - mean_val).powi(2)).sum::<f32>() / (data.len() - 1) as f32;

        Ok(variance.sqrt())
    }

    /// Clamp value between min and max
    pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }

    /// Linear interpolation between two values
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }
}

/// Feature detection utilities
pub mod features {
    /// Check if ONNX Runtime feature is enabled
    pub fn has_onnx_runtime() -> bool {
        cfg!(feature = "onnx-runtime")
    }

    /// Check if chat feature is enabled
    pub fn has_chat() -> bool {
        cfg!(feature = "chat")
    }

    /// Check if Jinja templates feature is enabled
    pub fn has_jinja_templates() -> bool {
        cfg!(feature = "jinja-templates")
    }

    /// Check if serde feature is enabled
    pub fn has_serde() -> bool {
        cfg!(feature = "serde")
    }

    /// Get list of all enabled features
    pub fn enabled_features() -> Vec<&'static str> {
        let mut features = Vec::new();

        if has_onnx_runtime() {
            features.push("onnx-runtime");
        }
        if has_chat() {
            features.push("chat");
        }
        if has_jinja_templates() {
            features.push("jinja-templates");
        }
        if has_serde() {
            features.push("serde");
        }

        features
    }
}

/// Environment and system utilities
pub mod env {
    use std::env;

    /// Get environment variable with default value
    pub fn get_env_or_default(key: &str, default: &str) -> String {
        env::var(key).unwrap_or_else(|_| default.to_string())
    }

    /// Get environment variable as integer with default
    pub fn get_env_int_or_default(key: &str, default: i32) -> i32 {
        env::var(key)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    /// Get environment variable as boolean with default
    pub fn get_env_bool_or_default(key: &str, default: bool) -> bool {
        env::var(key)
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            })
            .unwrap_or(default)
    }

    /// Check if running in debug mode
    pub fn is_debug_mode() -> bool {
        cfg!(debug_assertions) || get_env_bool_or_default("LIQUID_EDGE_DEBUG", false)
    }

    /// Get number of CPU cores
    pub fn cpu_count() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    /// Get optimal thread count for inference
    pub fn optimal_thread_count() -> usize {
        let cpu_count = cpu_count();
        let env_threads = get_env_int_or_default("LIQUID_EDGE_THREADS", 0);

        if env_threads > 0 {
            env_threads as usize
        } else {
            // Use all cores but cap at 8 for edge devices
            cpu_count.min(8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_string_truncate() {
        assert_eq!(string::truncate("hello", 10), "hello");
        assert_eq!(string::truncate("hello world", 8), "hello...");
        assert_eq!(string::truncate("hi", 2), "hi");
        assert_eq!(string::truncate("test", 3), "...");
    }

    #[test]
    fn test_bytes_to_human_readable() {
        assert_eq!(string::bytes_to_human_readable(0), "0 B");
        assert_eq!(string::bytes_to_human_readable(1024), "1.00 KB");
        assert_eq!(string::bytes_to_human_readable(1536), "1.50 KB");
        assert_eq!(string::bytes_to_human_readable(1048576), "1.00 MB");
    }

    #[test]
    fn test_validation() {
        assert!(validation::validate_token_id(5, 10).is_ok());
        assert!(validation::validate_token_id(10, 10).is_err());

        assert!(validation::validate_probability(0.5, "test").is_ok());
        assert!(validation::validate_probability(1.5, "test").is_err());

        assert!(validation::validate_temperature(0.0).is_ok());
        assert!(validation::validate_temperature(-1.0).is_err());
    }

    #[test]
    fn test_timer() {
        let timer = timing::Timer::start("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_conversion() {
        let input = vec![1u32, 2u32, 3u32];
        let converted = convert::u32_to_i64_slice(&input);
        assert_eq!(converted, vec![1i64, 2i64, 3i64]);

        let back_converted = convert::i64_to_u32_slice(&converted).unwrap();
        assert_eq!(back_converted, input);
    }

    #[test]
    fn test_math_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(math::mean(&data).unwrap(), 3.0);
        assert!(math::std_dev(&data).unwrap() > 0.0);

        assert_eq!(math::clamp(5, 0, 10), 5);
        assert_eq!(math::clamp(-1, 0, 10), 0);
        assert_eq!(math::clamp(15, 0, 10), 10);

        assert_eq!(math::lerp(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn test_fs_utilities() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").unwrap();

        assert!(super::fs::is_readable(&test_file));
        assert_eq!(super::fs::file_size(&test_file).unwrap(), 12);

        let content = super::fs::read_to_string_limited(&test_file, 100).unwrap();
        assert_eq!(content, "test content");

        assert!(super::fs::read_to_string_limited(&test_file, 5).is_err());
    }

    #[test]
    fn test_feature_detection() {
        let enabled = features::enabled_features();
        assert!(!enabled.is_empty()); // At least some features should be enabled

        // Test that we can detect our own features
        assert_eq!(features::has_onnx_runtime(), cfg!(feature = "onnx-runtime"));
    }

    #[test]
    fn test_env_utilities() {
        assert!(env::cpu_count() > 0);
        assert!(env::optimal_thread_count() > 0);

        // Test default values
        assert_eq!(
            env::get_env_or_default("NONEXISTENT_VAR", "default"),
            "default"
        );
        assert_eq!(env::get_env_int_or_default("NONEXISTENT_VAR", 42), 42);
        assert_eq!(env::get_env_bool_or_default("NONEXISTENT_VAR", true), true);
    }
}
