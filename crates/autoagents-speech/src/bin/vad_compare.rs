#[cfg(not(feature = "vad-compare"))]
fn main() {
    eprintln!("vad-compare binary requires: --features vad-compare");
}

#[cfg(feature = "vad-compare")]
mod compare {
    use autoagents_speech::ModelSource;
    use autoagents_speech::audio_capture::{AudioCapture, AudioCaptureConfig};
    use autoagents_speech::vad::{SileroVad, VadConfig, VadDevice, VadEngine, VadState};
    #[cfg(any(feature = "cuda", feature = "metal"))]
    use candle_core::DeviceLocation;
    use ndarray::{Array1, Array2, Array3};
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::value::Value;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    struct Args {
        audio_file: PathBuf,
        window_ms: u32,
        sample_rate: u32,
        max_windows: Option<usize>,
        print_every: usize,
        compare_state: bool,
        model_file: Option<PathBuf>,
        bench: bool,
        bench_only: bool,
        bench_runs: usize,
        bench_warmup: usize,
        bench_gpu_id: usize,
    }

    impl Args {
        fn parse() -> Result<Self, String> {
            let mut audio_file = None;
            let mut window_ms = 30;
            let mut sample_rate = 16_000;
            let mut max_windows = None;
            let mut print_every = 1usize;
            let mut compare_state = true;
            let mut model_file = None;
            let mut bench = false;
            let mut bench_only = false;
            let mut bench_runs = 3usize;
            let mut bench_warmup = 1usize;
            let mut bench_gpu_id = 0usize;

            let mut args = std::env::args().skip(1);
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--audio-file" => {
                        audio_file = Some(PathBuf::from(
                            args.next().ok_or("missing value for --audio-file")?,
                        ));
                    }
                    "--window-ms" => {
                        window_ms = args
                            .next()
                            .ok_or("missing value for --window-ms")?
                            .parse::<u32>()
                            .map_err(|_| "invalid --window-ms")?;
                    }
                    "--sample-rate" => {
                        sample_rate = args
                            .next()
                            .ok_or("missing value for --sample-rate")?
                            .parse::<u32>()
                            .map_err(|_| "invalid --sample-rate")?;
                    }
                    "--max-windows" => {
                        let value = args.next().ok_or("missing value for --max-windows")?;
                        max_windows = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| "invalid --max-windows")?,
                        );
                    }
                    "--print-every" => {
                        print_every = args
                            .next()
                            .ok_or("missing value for --print-every")?
                            .parse::<usize>()
                            .map_err(|_| "invalid --print-every")?;
                    }
                    "--no-state" => {
                        compare_state = false;
                    }
                    "--model-file" => {
                        model_file = Some(PathBuf::from(
                            args.next().ok_or("missing value for --model-file")?,
                        ));
                    }
                    "--bench" => {
                        bench = true;
                    }
                    "--bench-only" => {
                        bench = true;
                        bench_only = true;
                    }
                    "--bench-runs" => {
                        bench_runs = args
                            .next()
                            .ok_or("missing value for --bench-runs")?
                            .parse::<usize>()
                            .map_err(|_| "invalid --bench-runs")?;
                    }
                    "--bench-warmup" => {
                        bench_warmup = args
                            .next()
                            .ok_or("missing value for --bench-warmup")?
                            .parse::<usize>()
                            .map_err(|_| "invalid --bench-warmup")?;
                    }
                    "--bench-gpu-id" => {
                        bench_gpu_id = args
                            .next()
                            .ok_or("missing value for --bench-gpu-id")?
                            .parse::<usize>()
                            .map_err(|_| "invalid --bench-gpu-id")?;
                    }
                    other => return Err(format!("unknown argument: {other}")),
                }
            }

            let audio_file = audio_file.ok_or("provide --audio-file <path>")?;
            Ok(Self {
                audio_file,
                window_ms,
                sample_rate,
                max_windows,
                print_every: print_every.max(1),
                compare_state,
                model_file,
                bench,
                bench_only,
                bench_runs: bench_runs.max(1),
                bench_warmup,
                bench_gpu_id,
            })
        }
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse()?;

        let capture_config = AudioCaptureConfig::new(args.sample_rate, 1);
        let audio = AudioCapture::read_audio_with_config(&args.audio_file, capture_config)?;

        let model_source = match &args.model_file {
            Some(path) => ModelSource::from_file(path.clone()),
            None => ModelSource::from_hf("freddyaboulton/silero-vad", "silero_vad.onnx"),
        };
        let model_path = model_source.resolve()?;

        let window_samples =
            ((args.sample_rate as f32 * args.window_ms as f32) / 1000.0).round() as usize;
        if window_samples == 0 {
            return Err("window_ms must be > 0".into());
        }

        let windows_per_run = count_windows(&audio.samples, window_samples);
        if windows_per_run == 0 {
            return Err("audio does not contain a full window".into());
        }

        if !args.bench_only {
            let mut candle = SileroVad::new(model_source, VadConfig::new(args.sample_rate))?;
            let mut ort = OrtVad::new(&model_path, args.sample_rate)?;

            let mut max_prob_diff = 0.0_f32;
            let mut max_state_diff = 0.0_f32;
            let mut idx = 0usize;
            for chunk in audio.samples.chunks(window_samples) {
                if let Some(limit) = args.max_windows {
                    if idx >= limit {
                        break;
                    }
                }
                if chunk.len() < window_samples {
                    break;
                }

                let candle_out = candle.compute(chunk)?;
                let ort_out = ort.compute(chunk)?;

                let prob_diff = (candle_out.probability - ort_out.probability).abs();
                max_prob_diff = max_prob_diff.max(prob_diff);

                let mut state_diff = 0.0_f32;
                if args.compare_state {
                    let state = candle.state()?;
                    state_diff = max_state_delta(&state, &ort_out.state);
                    max_state_diff = max_state_diff.max(state_diff);
                }

                if idx % args.print_every == 0 {
                    if args.compare_state {
                        println!(
                            "window {idx:04}: candle={:.6} ort={:.6} diff={:.6} state_diff={:.6}",
                            candle_out.probability, ort_out.probability, prob_diff, state_diff
                        );
                    } else {
                        println!(
                            "window {idx:04}: candle={:.6} ort={:.6} diff={:.6}",
                            candle_out.probability, ort_out.probability, prob_diff
                        );
                    }
                }
                idx += 1;
            }

            println!(
                "done. windows={idx} max_prob_diff={:.6} max_state_diff={:.6}",
                max_prob_diff, max_state_diff
            );
        }

        if args.bench {
            run_benchmarks(
                &model_path,
                args.sample_rate,
                &audio.samples,
                window_samples,
                windows_per_run,
                args.bench_runs,
                args.bench_warmup,
                args.bench_gpu_id,
            )?;
        }
        Ok(())
    }

    fn run_benchmarks(
        model_path: &Path,
        sample_rate: u32,
        samples: &[f32],
        window_samples: usize,
        windows_per_run: usize,
        runs: usize,
        warmup: usize,
        gpu_id: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = gpu_id;
        println!(
            "bench: windows_per_run={windows_per_run} window_samples={window_samples} runs={runs} warmup={warmup}"
        );

        let cpu_config = VadConfig::new(sample_rate).with_device(VadDevice::Cpu);
        let mut candle_cpu = SileroVad::from_file(model_path, cpu_config)?;
        let cpu_result = bench_candle(
            &mut candle_cpu,
            "candle(cpu)",
            sample_rate,
            samples,
            window_samples,
            windows_per_run,
            runs,
            warmup,
        )?;
        print_bench(&cpu_result);

        #[cfg(feature = "cuda")]
        {
            let cuda_config = VadConfig::new(sample_rate).with_cuda_if_available(gpu_id);
            let mut candle_cuda = SileroVad::from_file(model_path, cuda_config)?;
            let location = candle_cuda.device_location();
            let label = match location {
                DeviceLocation::Cpu => "candle(cuda->cpu)".to_string(),
                _ => format!("candle({})", device_label(location)),
            };
            let cuda_result = bench_candle(
                &mut candle_cuda,
                &label,
                sample_rate,
                samples,
                window_samples,
                windows_per_run,
                runs,
                warmup,
            )?;
            print_bench(&cuda_result);
        }
        #[cfg(not(feature = "cuda"))]
        {
            println!("bench: candle(cuda) skipped (feature not enabled)");
        }

        #[cfg(feature = "metal")]
        {
            let metal_config = VadConfig::new(sample_rate).with_metal_if_available(gpu_id);
            let mut candle_metal = SileroVad::from_file(model_path, metal_config)?;
            let location = candle_metal.device_location();
            let label = match location {
                DeviceLocation::Cpu => "candle(metal->cpu)".to_string(),
                _ => format!("candle({})", device_label(location)),
            };
            let metal_result = bench_candle(
                &mut candle_metal,
                &label,
                sample_rate,
                samples,
                window_samples,
                windows_per_run,
                runs,
                warmup,
            )?;
            print_bench(&metal_result);
        }
        #[cfg(not(feature = "metal"))]
        {
            println!("bench: candle(metal) skipped (feature not enabled)");
        }

        let mut ort = OrtVad::new(model_path, sample_rate)?;
        let ort_result = bench_ort(
            &mut ort,
            "ort(cpu)",
            sample_rate,
            samples,
            window_samples,
            windows_per_run,
            runs,
            warmup,
        )?;
        print_bench(&ort_result);

        Ok(())
    }

    struct BenchResult {
        name: String,
        elapsed_s: f64,
        realtime: f64,
        windows_per_s: f64,
        runs: usize,
    }

    fn bench_candle(
        vad: &mut SileroVad,
        name: &str,
        sample_rate: u32,
        samples: &[f32],
        window_samples: usize,
        windows_per_run: usize,
        runs: usize,
        warmup: usize,
    ) -> Result<BenchResult, Box<dyn std::error::Error>> {
        for _ in 0..warmup {
            run_candle_pass(vad, samples, window_samples)?;
            vad.reset();
        }
        let start = Instant::now();
        for _ in 0..runs {
            run_candle_pass(vad, samples, window_samples)?;
            vad.reset();
        }
        let elapsed_s = start.elapsed().as_secs_f64();
        Ok(make_bench_result(
            name,
            sample_rate,
            window_samples,
            windows_per_run,
            runs,
            elapsed_s,
        ))
    }

    fn bench_ort(
        ort: &mut OrtVad,
        name: &str,
        sample_rate: u32,
        samples: &[f32],
        window_samples: usize,
        windows_per_run: usize,
        runs: usize,
        warmup: usize,
    ) -> Result<BenchResult, Box<dyn std::error::Error>> {
        for _ in 0..warmup {
            run_ort_pass(ort, samples, window_samples)?;
            ort.reset();
        }
        let start = Instant::now();
        for _ in 0..runs {
            run_ort_pass(ort, samples, window_samples)?;
            ort.reset();
        }
        let elapsed_s = start.elapsed().as_secs_f64();
        Ok(make_bench_result(
            name,
            sample_rate,
            window_samples,
            windows_per_run,
            runs,
            elapsed_s,
        ))
    }

    fn run_candle_pass(
        vad: &mut SileroVad,
        samples: &[f32],
        window_samples: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for chunk in samples.chunks(window_samples) {
            if chunk.len() < window_samples {
                break;
            }
            vad.compute(chunk)?;
        }
        Ok(())
    }

    fn run_ort_pass(
        ort: &mut OrtVad,
        samples: &[f32],
        window_samples: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for chunk in samples.chunks(window_samples) {
            if chunk.len() < window_samples {
                break;
            }
            ort.compute(chunk)?;
        }
        Ok(())
    }

    fn make_bench_result(
        name: &str,
        sample_rate: u32,
        window_samples: usize,
        windows_per_run: usize,
        runs: usize,
        elapsed_s: f64,
    ) -> BenchResult {
        let total_windows = windows_per_run * runs;
        let audio_seconds = (total_windows * window_samples) as f64 / sample_rate as f64;
        let realtime = audio_seconds / elapsed_s;
        let windows_per_s = total_windows as f64 / elapsed_s;
        BenchResult {
            name: name.to_string(),
            elapsed_s,
            realtime,
            windows_per_s,
            runs,
        }
    }

    fn print_bench(result: &BenchResult) {
        println!(
            "bench: {:<16} elapsed={:.3}s realtime={:.2}x windows/s={:.1} runs={}",
            result.name, result.elapsed_s, result.realtime, result.windows_per_s, result.runs
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn device_label(location: DeviceLocation) -> String {
        match location {
            DeviceLocation::Cpu => "cpu".to_string(),
            DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
        }
    }

    fn count_windows(samples: &[f32], window_samples: usize) -> usize {
        if window_samples == 0 {
            return 0;
        }
        let mut count = 0usize;
        for chunk in samples.chunks(window_samples) {
            if chunk.len() < window_samples {
                break;
            }
            count += 1;
        }
        count
    }

    fn max_state_delta(candle: &VadState, ort: &VadState) -> f32 {
        let mut max = 0.0_f32;
        for (a, b) in candle.h.iter().zip(ort.h.iter()) {
            let diff = (*a - *b).abs();
            max = max.max(diff);
        }
        for (a, b) in candle.c.iter().zip(ort.c.iter()) {
            let diff = (*a - *b).abs();
            max = max.max(diff);
        }
        max
    }

    struct OrtVad {
        session: Session,
        h: Array3<f32>,
        c: Array3<f32>,
        sample_rate_i64: i64,
    }

    struct OrtOutput {
        probability: f32,
        state: VadState,
    }

    impl OrtVad {
        fn new(path: &Path, sample_rate: u32) -> Result<Self, Box<dyn std::error::Error>> {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .with_inter_threads(1)?
                .commit_from_file(path)?;

            Ok(Self {
                session,
                h: Array3::<f32>::zeros((2, 1, 64)),
                c: Array3::<f32>::zeros((2, 1, 64)),
                sample_rate_i64: sample_rate as i64,
            })
        }

        fn compute(&mut self, samples: &[f32]) -> Result<OrtOutput, Box<dyn std::error::Error>> {
            let samples_tensor = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;
            let samples_value = Value::from_array(samples_tensor)?;
            let sr_value = Value::from_array(Array1::from_elem(1, self.sample_rate_i64))?;
            let h_value = Value::from_array(self.h.clone())?;
            let c_value = Value::from_array(self.c.clone())?;

            let result = self.session.run(ort::inputs![
                "input" => samples_value,
                "sr" => sr_value,
                "h" => h_value,
                "c" => c_value
            ])?;

            let h_output = result
                .get("hn")
                .ok_or("missing output 'hn'")?
                .try_extract_tensor::<f32>()?;
            self.h = Array3::from_shape_vec((2, 1, 64), h_output.1.to_vec())?;

            let c_output = result
                .get("cn")
                .ok_or("missing output 'cn'")?
                .try_extract_tensor::<f32>()?;
            self.c = Array3::from_shape_vec((2, 1, 64), c_output.1.to_vec())?;

            let output = result
                .get("output")
                .ok_or("missing output 'output'")?
                .try_extract_tensor::<f32>()?;
            let prob = output.1.first().copied().unwrap_or(0.0);

            Ok(OrtOutput {
                probability: prob,
                state: VadState {
                    h: h_output.1.to_vec(),
                    c: c_output.1.to_vec(),
                },
            })
        }

        fn reset(&mut self) {
            self.h.fill(0.0);
            self.c.fill(0.0);
        }
    }
}

#[cfg(feature = "vad-compare")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    compare::run()
}
