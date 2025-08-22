use anyhow::{Error as E, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{distr::Distribution, SeedableRng};
use tokenizers::Tokenizer;
use candle_transformers::models::whisper::{self as m, audio, Config};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::stt::{token_id, Decoder, Model, Task, WhichModel};

mod multilingual;
mod stt;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    quantized: bool,

    /// Language.
    #[arg(long)]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    task: Option<Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,

    /// The input device to use.
    #[arg(long)]
    device: Option<String>,
}

fn run_stt() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let (default_model, default_revision) = if args.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    let (model_id, revision) = match (args.model_id, args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let (config, tokenizer, model) = if args.quantized {
            let ext = match args.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", args.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let model = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_filename,
            &device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
    } else {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };
    let mut decoder = Decoder::new(
        model,
        tokenizer.clone(),
        args.seed,
        &device,
        /* language_token */ None,
        args.task,
        args.timestamps,
        args.verbose,
    )?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("../melfilters.bytes").as_slice(),
        128 => include_bytes!("../melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    // Set up the input device and stream with the default input config.
    let host = cpal::default_host();
    let audio_device = match args.device.as_ref() {
        None => host.default_input_device(),
        Some(device) => host
            .input_devices()?
            .find(|x| x.name().map_or(false, |y| &y == device)),
    }
        .expect("failed to find the audio input device");

    let audio_config = audio_device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("audio config {audio_config:?}");

    let channel_count = audio_config.channels() as usize;
    let in_sample_rate = audio_config.sample_rate().0 as usize;
    let resample_ratio = 16000. / in_sample_rate as f64;
    let mut resampler = rubato::FastFixedIn::new(
        resample_ratio,
        10.,
        rubato::PolynomialDegree::Septic,
        1024,
        1,
    )?;
    let (tx, rx) = std::sync::mpsc::channel();
    let stream = audio_device.build_input_stream(
        &audio_config.config(),
        move |pcm: &[f32], _: &cpal::InputCallbackInfo| {
            let pcm = pcm
                .iter()
                .step_by(channel_count)
                .copied()
                .collect::<Vec<f32>>();
            if !pcm.is_empty() {
                tx.send(pcm).unwrap()
            }
        },
        move |err| {
            eprintln!("an error occurred on stream: {}", err);
        },
        None,
    )?;
    stream.play()?;

    // loop to process the audio data forever (until the user stops the program)
    println!("transcribing audio...");
    let mut buffered_pcm = vec![];
    let mut language_token_set = false;
    while let Ok(pcm) = rx.recv() {
        use rubato::Resampler;

        buffered_pcm.extend_from_slice(&pcm);
        if buffered_pcm.len() < 10 * in_sample_rate {
            continue;
        }
        let mut resampled_pcm = vec![];
        println!("Processing");
        // resample the audio, one chunk of 1024 samples at a time.
        // in case the audio input failed to produce an exact multiple of 1024 samples,
        // process the remainder on the next iteration of the loop.
        let full_chunks = buffered_pcm.len() / 1024;
        let remainder = buffered_pcm.len() % 1024;
        for chunk in 0..full_chunks {
            let buffered_pcm = &buffered_pcm[chunk * 1024..(chunk + 1) * 1024];
            let pcm = resampler.process(&[&buffered_pcm], None)?;
            resampled_pcm.extend_from_slice(&pcm[0]);
        }
        let pcm = resampled_pcm;
        println!("{} {}", buffered_pcm.len(), pcm.len());
        if remainder == 0 {
            buffered_pcm.clear();
        } else {
            // efficiently copy the remainder to the beginning of the `buffered_pcm` buffer and
            // truncate it.  That's more efficient then allocating a new vector and copying into it
            println!("audio device produced partial chunk with {remainder} samples; processing the remainder on the next iteration of the loop");
            buffered_pcm.copy_within(full_chunks * 1024.., 0);
            buffered_pcm.truncate(remainder);
        }
        let mel = audio::pcm_to_mel(&config, &pcm, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        )?;

        // on the first iteration, we detect the language and set the language token.
        if !language_token_set {
            let language_token = match (args.model.is_multilingual(), args.language.clone()) {
                (true, None) => Some(multilingual::detect_language(
                    decoder.model(),
                    &tokenizer,
                    &mel,
                )?),
                (false, None) => None,
                (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
                    Ok(token_id) => Some(token_id),
                    Err(_) => anyhow::bail!("language {language} is not supported"),
                },
                (false, Some(_)) => {
                    anyhow::bail!("a language cannot be set for non-multilingual models")
                }
            };
            println!("language_token: {:?}", language_token);
            decoder.set_language_token(language_token);
            language_token_set = true;
        }
        decoder.run(&mel, None)?;
        decoder.reset_kv_cache();
    }

    Ok(())
}