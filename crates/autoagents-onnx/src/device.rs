//! Device abstraction for liquid-edge inference
//!
//! This module provides device abstractions following the USLS pattern.
//! Devices are simple enums that can be converted to ORT execution providers.

use serde::{Deserialize, Serialize};
use std::fmt;

#[allow(unused_imports)]
use ort::execution_providers::ExecutionProvider;

/// Device types for model execution, following USLS pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Device {
    /// CPU device with thread count
    Cpu(usize),
    /// CUDA device with device ID
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        {
            Self::Cpu(0)
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(i) => write!(f, "cpu:{i}"),
            #[cfg(feature = "cuda")]
            Self::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = crate::EdgeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        #[inline]
        fn parse_device_id(id_str: Option<&str>) -> usize {
            id_str
                .map(|s| s.trim().parse::<usize>().unwrap_or(0))
                .unwrap_or(0)
        }

        let (device_type, id_part) = s
            .trim()
            .split_once(':')
            .map_or_else(|| (s.trim(), None), |(device, id)| (device, Some(id)));

        match device_type.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu(parse_device_id(id_part))),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(Self::Cuda(parse_device_id(id_part))),
            _ => Err(crate::EdgeError::runtime(format!(
                "Unsupported device: {s}"
            ))),
        }
    }
}

impl Device {
    /// Get the device ID if applicable
    pub fn id(&self) -> Option<usize> {
        match self {
            Self::Cpu(i) => Some(*i),
            #[cfg(feature = "cuda")]
            Self::Cuda(i) => Some(*i),
        }
    }

    /// Check if the device is available on the system
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cpu(_) => true, // CPU is always available
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => {
                use ort::execution_providers::CUDAExecutionProvider;
                CUDAExecutionProvider::default()
                    .with_device_id(self.id().unwrap_or(0) as i32)
                    .is_available()
                    .unwrap_or(false)
            }
        }
    }
}

/// Convenience functions for device creation
pub fn cpu() -> Device {
    Device::Cpu(1)
}

pub fn cpu_with_threads(threads: usize) -> Device {
    Device::Cpu(threads)
}

#[cfg(feature = "cuda")]
pub fn cuda(device_id: usize) -> Device {
    Device::Cuda(device_id)
}

#[cfg(feature = "cuda")]
pub fn cuda_default() -> Device {
    Device::Cuda(0)
}
