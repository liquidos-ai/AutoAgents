pub type ElemType = f32;
pub const DTYPE_NAME: &str = "f32";

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::wgpu::{WebGpu, WgpuDevice};
    pub type InferenceBackend = WebGpu<ElemType>;
    pub type InferenceDevice = WgpuDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = WgpuDevice::DefaultDevice;
    pub const NAME: &str = "webgpu";
}

#[cfg(feature = "candle-metal")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle<ElemType>;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CandleDevice> =
        std::sync::LazyLock::new(|| CandleDevice::metal(0));
    pub const NAME: &str = "candle-metal";
}

#[cfg(feature = "candle-cpu")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle<ElemType>;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CandleDevice> =
        std::sync::LazyLock::new(|| CandleDevice::Cpu);
    pub const NAME: &str = "candle-cpu";
}

#[cfg(feature = "ndarray")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub type InferenceBackend = NdArray<ElemType>;
    pub type InferenceDevice = NdArrayDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = NdArrayDevice::Cpu;
    pub const NAME: &str = "ndarray";
}

#[cfg(all(feature = "rocm", not(target_arch = "wasm32")))]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::rocm::{Rocm, RocmDevice};
    pub type InferenceBackend = Rocm<ElemType>;
    pub type InferenceDevice = RocmDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<RocmDevice> =
        std::sync::LazyLock::new(|| RocmDevice::default());
    pub const NAME: &str = "rocm";
}

#[cfg(all(
    feature = "cuda",
    not(feature = "ndarray"),
    not(target_arch = "wasm32"),
))]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};
    pub type InferenceBackend = Cuda<ElemType>;
    pub type InferenceDevice = CudaDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CudaDevice> =
        std::sync::LazyLock::new(|| CudaDevice::default());
    pub const NAME: &str = "cuda";
}
