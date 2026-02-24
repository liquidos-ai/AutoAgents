use super::VadEngine;
use super::config::{VadConfig, VadDevice};
use super::error::{VadError, VadResult};
use super::result::VadOutput;
use crate::model_source::ModelSource;
use candle_core::{DType, Device, DeviceLocation, IndexOp, Tensor};
use candle_onnx::eval::get_tensor;
use candle_onnx::onnx::{self, attribute_proto::AttributeType};
use candle_onnx::read_file;
use std::collections::HashMap;
use std::path::Path;

const PAD_SAMPLES: usize = 96;
const STFT_STRIDE: usize = 64;
const STFT_BINS: usize = 129;
const MAG_SCALE: f32 = 1_048_576.0;
const MAG_BIAS: f32 = 1.0;

struct Conv1dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl Conv1dLayer {
    fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            groups,
        }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.conv1d_with_algo(
            &self.weight,
            self.padding,
            self.stride,
            1,
            self.groups,
            None,
        )?;
        match &self.bias {
            None => Ok(y),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                y.broadcast_add(&bias)
            }
        }
    }
}

struct ConvBlock {
    depthwise: Conv1dLayer,
    pointwise: Conv1dLayer,
    projection: Option<Conv1dLayer>,
}

impl ConvBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.depthwise.forward(x)?.relu()?;
        let y = self.pointwise.forward(&y)?;
        let residual = match &self.projection {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };
        let y = y.broadcast_add(&residual)?;
        y.relu()
    }
}

struct LstmWeights {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
}

impl LstmWeights {
    fn from_onnx(w: Tensor, r: Tensor, b: Tensor) -> VadResult<Self> {
        let w = w.squeeze(0).map_err(to_model_load)?;
        let r = r.squeeze(0).map_err(to_model_load)?;
        let b = b.squeeze(0).map_err(to_model_load)?;
        let hidden = w.dim(0).map_err(to_model_load)? / 4;
        if hidden == 0 {
            return Err(VadError::ModelLoad("invalid LSTM hidden size".to_string()));
        }
        let b_ih = b.narrow(0, 0, hidden * 4).map_err(to_model_load)?;
        let b_hh = b.narrow(0, hidden * 4, hidden * 4).map_err(to_model_load)?;
        Ok(Self {
            w_ih: w,
            w_hh: r,
            b_ih,
            b_hh,
        })
    }
}

struct SileroBranch {
    device: Device,
    feature_extractor: Conv1dLayer,
    adaptive_filter: Conv1dLayer,
    first_block: ConvBlock,
    down1: Conv1dLayer,
    enc3: ConvBlock,
    down2: Conv1dLayer,
    enc7: ConvBlock,
    down3: Conv1dLayer,
    enc11: ConvBlock,
    post: Conv1dLayer,
    lstm1: LstmWeights,
    lstm2: LstmWeights,
    decoder: Conv1dLayer,
    mag_scale: Tensor,
    mag_bias: Tensor,
    one: Tensor,
}

impl SileroBranch {
    fn forward(&self, samples: &[f32], h: &Tensor, c: &Tensor) -> VadResult<(f32, Tensor, Tensor)> {
        let input =
            Tensor::from_slice(samples, (1, samples.len()), &self.device).map_err(to_inference)?;
        let input = input.unsqueeze(1).map_err(to_inference)?;
        let input = reflect_pad_1d(&input, PAD_SAMPLES).map_err(to_inference)?;

        let forward = self
            .feature_extractor
            .forward(&input)
            .map_err(to_inference)?;
        let real = forward.narrow(1, 0, STFT_BINS).map_err(to_inference)?;
        let imag = forward
            .narrow(1, STFT_BINS, STFT_BINS)
            .map_err(to_inference)?;
        let magnitude = real
            .sqr()
            .map_err(to_inference)?
            .broadcast_add(&imag.sqr().map_err(to_inference)?)
            .map_err(to_inference)?
            .sqrt()
            .map_err(to_inference)?;
        let spect = magnitude
            .broadcast_mul(&self.mag_scale)
            .map_err(to_inference)?
            .broadcast_add(&self.mag_bias)
            .map_err(to_inference)?
            .log()
            .map_err(to_inference)?;

        let mean = spect.mean_keepdim(1).map_err(to_inference)?;
        let frames = mean.dim(2).map_err(to_inference)?;
        if frames < 4 {
            return Err(VadError::Inference(format!(
                "insufficient frames for VAD normalization: {frames}"
            )));
        }
        let left = mean
            .narrow(2, 1, 3)
            .map_err(to_inference)?
            .flip(&[2])
            .map_err(to_inference)?;
        let right = mean
            .narrow(2, frames - 4, 3)
            .map_err(to_inference)?
            .flip(&[2])
            .map_err(to_inference)?;
        let mean0 = Tensor::cat(&[left, mean.clone(), right], 2).map_err(to_inference)?;
        let mean1 = self.adaptive_filter.forward(&mean0).map_err(to_inference)?;
        let mean_mean = mean1.mean_keepdim(2).map_err(to_inference)?;
        let norm = spect
            .broadcast_add(&mean_mean.neg().map_err(to_inference)?)
            .map_err(to_inference)?;

        let x1 = Tensor::cat(&[magnitude, norm], 1).map_err(to_inference)?;
        let x = self.first_block.forward(&x1).map_err(to_inference)?;
        let x = self
            .down1
            .forward(&x)
            .map_err(to_inference)?
            .relu()
            .map_err(to_inference)?;
        let x = self.enc3.forward(&x).map_err(to_inference)?;
        let x = self
            .down2
            .forward(&x)
            .map_err(to_inference)?
            .relu()
            .map_err(to_inference)?;
        let x = self.enc7.forward(&x).map_err(to_inference)?;
        let x = self
            .down3
            .forward(&x)
            .map_err(to_inference)?
            .relu()
            .map_err(to_inference)?;
        let x = self.enc11.forward(&x).map_err(to_inference)?;
        let x = self
            .post
            .forward(&x)
            .map_err(to_inference)?
            .relu()
            .map_err(to_inference)?;

        let seq = x.permute((2, 0, 1)).map_err(to_inference)?;
        let h0 = h
            .narrow(0, 0, 1)
            .map_err(to_inference)?
            .squeeze(0)
            .map_err(to_inference)?;
        let c0 = c
            .narrow(0, 0, 1)
            .map_err(to_inference)?
            .squeeze(0)
            .map_err(to_inference)?;
        let h1 = h
            .narrow(0, 1, 1)
            .map_err(to_inference)?
            .squeeze(0)
            .map_err(to_inference)?;
        let c1 = c
            .narrow(0, 1, 1)
            .map_err(to_inference)?
            .squeeze(0)
            .map_err(to_inference)?;

        let (l1_out, h0_new, c0_new) =
            lstm_seq(&seq, &h0, &c0, &self.lstm1, &self.one).map_err(to_inference)?;
        let (l2_out, h1_new, c1_new) =
            lstm_seq(&l1_out, &h1, &c1, &self.lstm2, &self.one).map_err(to_inference)?;
        let lstm_out = l2_out.permute((1, 2, 0)).map_err(to_inference)?;
        let lstm_out = lstm_out.relu().map_err(to_inference)?;

        let logits = self.decoder.forward(&lstm_out).map_err(to_inference)?;
        let probs = sigmoid(&logits, &self.one).map_err(to_inference)?;
        let prob = probs
            .mean((1, 2))
            .map_err(to_inference)?
            .to_vec1::<f32>()
            .map_err(to_inference)?
            .first()
            .copied()
            .unwrap_or(0.0);

        let new_h = Tensor::stack(&[h0_new, h1_new], 0).map_err(to_inference)?;
        let new_c = Tensor::stack(&[c0_new, c1_new], 0).map_err(to_inference)?;

        Ok((prob, new_h, new_c))
    }
}

fn reflect_pad_1d(input: &Tensor, pad: usize) -> candle_core::Result<Tensor> {
    if pad == 0 {
        return Ok(input.clone());
    }
    let len = input.dim(2)?;
    if len <= pad {
        return Err(candle_core::Error::msg(format!(
            "reflect pad requires length > pad (len={len}, pad={pad})"
        )));
    }
    let left = input.narrow(2, 1, pad)?.flip(&[2])?;
    let right = input.narrow(2, len - pad - 1, pad)?.flip(&[2])?;
    Tensor::cat(&[left, input.clone(), right], 2)
}

fn lstm_seq(
    input: &Tensor,
    h0: &Tensor,
    c0: &Tensor,
    weights: &LstmWeights,
    one: &Tensor,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let (seq_len, _batch, _features) = input.dims3()?;
    let mut outputs = Vec::with_capacity(seq_len);
    let mut h = h0.clone();
    let mut c = c0.clone();
    for step in 0..seq_len {
        let x = input.i((step, .., ..))?.contiguous()?;
        let (next_h, next_c) = lstm_step(&x, &h, &c, weights, one)?;
        outputs.push(next_h.clone());
        h = next_h;
        c = next_c;
    }
    let out = Tensor::stack(&outputs, 0)?;
    Ok((out, h, c))
}

fn lstm_step(
    x: &Tensor,
    h: &Tensor,
    c: &Tensor,
    weights: &LstmWeights,
    one: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let w_ih = x.matmul(&weights.w_ih.t()?)?;
    let w_hh = h.matmul(&weights.w_hh.t()?)?;
    let w_ih = w_ih.broadcast_add(&weights.b_ih)?;
    let w_hh = w_hh.broadcast_add(&weights.b_hh)?;
    let gates = w_ih.broadcast_add(&w_hh)?;
    let chunks = gates.chunk(4, 1)?;

    let input_gate = sigmoid(&chunks[0], one)?;
    let output_gate = sigmoid(&chunks[1], one)?;
    let forget_gate = sigmoid(&chunks[2], one)?;
    let cell_gate = chunks[3].tanh()?;

    let next_c = forget_gate
        .mul(c)?
        .broadcast_add(&input_gate.mul(&cell_gate)?)?;
    let next_h = output_gate.mul(&next_c.tanh()?)?;
    Ok((next_h, next_c))
}

fn sigmoid(x: &Tensor, one: &Tensor) -> candle_core::Result<Tensor> {
    let denom = x.neg()?.exp()?.broadcast_add(one)?;
    denom.recip()
}

struct SileroModel {
    branch_16k: SileroBranch,
    branch_8k: SileroBranch,
}

impl SileroModel {
    fn load(path: &Path, device: &Device) -> VadResult<Self> {
        let model = read_file(path).map_err(to_model_load)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| VadError::ModelLoad("missing ONNX graph".to_string()))?;
        let mut initializers = HashMap::new();
        collect_initializers(graph, &mut initializers, device)?;

        let branch_16k = build_branch(
            BranchSpec {
                prefix: "model",
                down1: ("1110", "1111", 2),
                down2: ("1113", "1114", 2),
                down3: ("1116", "1117", 2),
                post: ("1119", "1120", 1),
                lstm1: ("343", "345", "347"),
                lstm2: ("415", "417", "419"),
            },
            &initializers,
            device,
        )?;
        let branch_8k = build_branch(
            BranchSpec {
                prefix: "model_8k",
                down1: ("1122", "1123", 2),
                down2: ("1125", "1126", 2),
                down3: ("1128", "1129", 1),
                post: ("1131", "1132", 1),
                lstm1: ("833", "835", "837"),
                lstm2: ("905", "907", "909"),
            },
            &initializers,
            device,
        )?;

        Ok(Self {
            branch_16k,
            branch_8k,
        })
    }
}

struct BranchSpec<'a> {
    prefix: &'a str,
    down1: (&'a str, &'a str, usize),
    down2: (&'a str, &'a str, usize),
    down3: (&'a str, &'a str, usize),
    post: (&'a str, &'a str, usize),
    lstm1: (&'a str, &'a str, &'a str),
    lstm2: (&'a str, &'a str, &'a str),
}

fn build_branch(
    spec: BranchSpec<'_>,
    inits: &HashMap<String, Tensor>,
    device: &Device,
) -> VadResult<SileroBranch> {
    let feature_extractor = Conv1dLayer::new(
        get_init(
            inits,
            &format!("{}.feature_extractor.forward_basis_buffer", spec.prefix),
        )?,
        None,
        STFT_STRIDE,
        0,
        1,
    );
    let adaptive_filter = Conv1dLayer::new(
        get_init(
            inits,
            &format!("{}.adaptive_normalization.filter_", spec.prefix),
        )?,
        None,
        1,
        0,
        1,
    );
    let first_block = ConvBlock {
        depthwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.first_layer.0.dw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.first_layer.0.dw_conv.0.bias", spec.prefix),
            )?),
            1,
            2,
            STFT_BINS * 2,
        ),
        pointwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.first_layer.0.pw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.first_layer.0.pw_conv.0.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        ),
        projection: Some(Conv1dLayer::new(
            get_init(inits, &format!("{}.first_layer.0.proj.weight", spec.prefix))?,
            Some(get_init(
                inits,
                &format!("{}.first_layer.0.proj.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        )),
    };

    let down1 = Conv1dLayer::new(
        get_init(inits, spec.down1.0)?,
        Some(get_init(inits, spec.down1.1)?),
        spec.down1.2,
        0,
        1,
    );
    let enc3 = ConvBlock {
        depthwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.encoder.3.0.dw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.encoder.3.0.dw_conv.0.bias", spec.prefix),
            )?),
            1,
            2,
            16,
        ),
        pointwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.encoder.3.0.pw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.encoder.3.0.pw_conv.0.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        ),
        projection: Some(Conv1dLayer::new(
            get_init(inits, &format!("{}.encoder.3.0.proj.weight", spec.prefix))?,
            Some(get_init(
                inits,
                &format!("{}.encoder.3.0.proj.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        )),
    };
    let down2 = Conv1dLayer::new(
        get_init(inits, spec.down2.0)?,
        Some(get_init(inits, spec.down2.1)?),
        spec.down2.2,
        0,
        1,
    );
    let enc7 = ConvBlock {
        depthwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.encoder.7.0.dw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.encoder.7.0.dw_conv.0.bias", spec.prefix),
            )?),
            1,
            2,
            32,
        ),
        pointwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.encoder.7.0.pw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.encoder.7.0.pw_conv.0.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        ),
        projection: None,
    };
    let down3 = Conv1dLayer::new(
        get_init(inits, spec.down3.0)?,
        Some(get_init(inits, spec.down3.1)?),
        spec.down3.2,
        0,
        1,
    );
    let enc11 = ConvBlock {
        depthwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.encoder.11.0.dw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.encoder.11.0.dw_conv.0.bias", spec.prefix),
            )?),
            1,
            2,
            32,
        ),
        pointwise: Conv1dLayer::new(
            get_init(
                inits,
                &format!("{}.encoder.11.0.pw_conv.0.weight", spec.prefix),
            )?,
            Some(get_init(
                inits,
                &format!("{}.encoder.11.0.pw_conv.0.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        ),
        projection: Some(Conv1dLayer::new(
            get_init(inits, &format!("{}.encoder.11.0.proj.weight", spec.prefix))?,
            Some(get_init(
                inits,
                &format!("{}.encoder.11.0.proj.bias", spec.prefix),
            )?),
            1,
            0,
            1,
        )),
    };
    let post = Conv1dLayer::new(
        get_init(inits, spec.post.0)?,
        Some(get_init(inits, spec.post.1)?),
        spec.post.2,
        0,
        1,
    );
    let lstm1 = LstmWeights::from_onnx(
        get_init(inits, spec.lstm1.0)?,
        get_init(inits, spec.lstm1.1)?,
        get_init(inits, spec.lstm1.2)?,
    )?;
    let lstm2 = LstmWeights::from_onnx(
        get_init(inits, spec.lstm2.0)?,
        get_init(inits, spec.lstm2.1)?,
        get_init(inits, spec.lstm2.2)?,
    )?;
    let decoder = Conv1dLayer::new(
        get_init(inits, &format!("{}.decoder.decoder.1.weight", spec.prefix))?,
        Some(get_init(
            inits,
            &format!("{}.decoder.decoder.1.bias", spec.prefix),
        )?),
        1,
        0,
        1,
    );

    Ok(SileroBranch {
        device: device.clone(),
        feature_extractor,
        adaptive_filter,
        first_block,
        down1,
        enc3,
        down2,
        enc7,
        down3,
        enc11,
        post,
        lstm1,
        lstm2,
        decoder,
        mag_scale: Tensor::from_slice(&[MAG_SCALE], (), device).map_err(to_model_load)?,
        mag_bias: Tensor::from_slice(&[MAG_BIAS], (), device).map_err(to_model_load)?,
        one: Tensor::from_slice(&[1f32], (), device).map_err(to_model_load)?,
    })
}

fn get_init(inits: &HashMap<String, Tensor>, name: &str) -> VadResult<Tensor> {
    inits
        .get(name)
        .cloned()
        .ok_or_else(|| VadError::ModelLoad(format!("missing initializer '{name}'")))
}

fn collect_initializers(
    graph: &onnx::GraphProto,
    out: &mut HashMap<String, Tensor>,
    device: &Device,
) -> VadResult<()> {
    for init in &graph.initializer {
        if out.contains_key(init.name.as_str()) {
            continue;
        }
        let tensor = get_tensor(init, init.name.as_str()).map_err(to_model_load)?;
        let tensor = tensor.to_device(device).map_err(to_model_load)?;
        out.insert(init.name.clone(), tensor);
    }
    for node in &graph.node {
        if node.op_type == "If" {
            for attr in &node.attribute {
                if attr.r#type == AttributeType::Graph as i32
                    && let Some(g) = &attr.g
                {
                    collect_initializers(g, out, device)?;
                }
            }
        }
    }
    Ok(())
}

fn to_model_load(err: impl std::fmt::Display) -> VadError {
    VadError::ModelLoad(err.to_string())
}

fn to_inference(err: impl std::fmt::Display) -> VadError {
    VadError::Inference(err.to_string())
}

fn resolve_device(config: &VadConfig) -> VadResult<Device> {
    match config.device {
        VadDevice::Cpu => Ok(Device::Cpu),
        VadDevice::Cuda { gpu_id } => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(gpu_id).map_err(to_model_load)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = gpu_id;
                Err(VadError::ModelLoad(
                    "cuda feature not enabled for autoagents-speech".to_string(),
                ))
            }
        }
        VadDevice::Metal { gpu_id } => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(gpu_id).map_err(to_model_load)
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = gpu_id;
                Err(VadError::ModelLoad(
                    "metal feature not enabled for autoagents-speech".to_string(),
                ))
            }
        }
        VadDevice::CudaIfAvailable { gpu_id } => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda_if_available(gpu_id).map_err(to_model_load)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = gpu_id;
                Ok(Device::Cpu)
            }
        }
        VadDevice::MetalIfAvailable { gpu_id } => {
            #[cfg(feature = "metal")]
            {
                Device::metal_if_available(gpu_id).map_err(to_model_load)
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = gpu_id;
                Ok(Device::Cpu)
            }
        }
    }
}

/// Silero VAD engine backed by a native Candle model.
pub struct SileroVad {
    model: SileroModel,
    h_tensor: Tensor,
    c_tensor: Tensor,
    sample_rate: u32,
}

impl SileroVad {
    pub fn new(model_source: ModelSource, config: VadConfig) -> VadResult<Self> {
        config.validate()?;
        let model_path = model_source.resolve()?;
        let device = resolve_device(&config)?;
        let model = SileroModel::load(&model_path, &device)?;
        let h_tensor = Tensor::zeros((2, 1, 64), DType::F32, &device).map_err(to_model_load)?;
        let c_tensor = Tensor::zeros((2, 1, 64), DType::F32, &device).map_err(to_model_load)?;

        Ok(Self {
            model,
            h_tensor,
            c_tensor,
            sample_rate: config.sample_rate,
        })
    }

    pub fn from_file(path: impl Into<std::path::PathBuf>, config: VadConfig) -> VadResult<Self> {
        Self::new(ModelSource::from_file(path), config)
    }

    pub fn from_hf(
        repo_id: impl Into<String>,
        filename: impl Into<String>,
        config: VadConfig,
    ) -> VadResult<Self> {
        Self::new(ModelSource::from_hf(repo_id, filename), config)
    }

    pub fn reset(&mut self) {
        if let Ok(h) = self.h_tensor.zeros_like() {
            self.h_tensor = h;
        }
        if let Ok(c) = self.c_tensor.zeros_like() {
            self.c_tensor = c;
        }
    }

    pub fn device_location(&self) -> DeviceLocation {
        self.h_tensor.device().location()
    }
}

#[cfg(feature = "vad-compare")]
#[derive(Debug, Clone)]
pub struct VadState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

#[cfg(feature = "vad-compare")]
impl SileroVad {
    pub fn state(&self) -> VadResult<VadState> {
        let h = self
            .h_tensor
            .flatten_all()
            .map_err(to_inference)?
            .to_vec1::<f32>()
            .map_err(to_inference)?;
        let c = self
            .c_tensor
            .flatten_all()
            .map_err(to_inference)?
            .to_vec1::<f32>()
            .map_err(to_inference)?;
        Ok(VadState { h, c })
    }
}

impl VadEngine for SileroVad {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn compute(&mut self, samples: &[f32]) -> VadResult<VadOutput> {
        if samples.is_empty() {
            return Err(VadError::InvalidInput(
                "VAD input must contain at least one sample".to_string(),
            ));
        }

        let (prob, new_h, new_c) = match self.sample_rate {
            8_000 => self
                .model
                .branch_8k
                .forward(samples, &self.h_tensor, &self.c_tensor)?,
            16_000 => self
                .model
                .branch_16k
                .forward(samples, &self.h_tensor, &self.c_tensor)?,
            other => return Err(VadError::UnsupportedSampleRate(other)),
        };

        self.h_tensor = new_h;
        self.c_tensor = new_c;

        Ok(VadOutput { probability: prob })
    }
}
