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

#[cfg(feature = "ndarray")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub type InferenceBackend = NdArray<ElemType>;
    pub type InferenceDevice = NdArrayDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = NdArrayDevice::Cpu;
    pub const NAME: &str = "ndarray";
}

#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};
    pub type InferenceBackend = Cuda<ElemType>;
    pub type InferenceDevice = CudaDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CudaDevice> =
        std::sync::LazyLock::new(|| CudaDevice::default());
    pub const NAME: &str = "cuda";
}
