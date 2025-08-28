async function fetchArrayBuffer(url) {
  const cacheName = "phi-mixformer-candle-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}

async function concatenateArrayBuffers(urls) {
  const arrayBuffers = await Promise.all(urls.map(url => fetchArrayBuffer(url)));

  let totalLength = arrayBuffers.reduce((acc, arrayBuffer) => acc + arrayBuffer.byteLength, 0);
  let concatenatedBuffer = new Uint8Array(totalLength);

  let offset = 0;
  arrayBuffers.forEach(buffer => {
    concatenatedBuffer.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  });
  return concatenatedBuffer;
}

class PhiModelLoader {
  static async loadModelFiles(weightsURL, tokenizerURL, configURL) {
    try {
      self.postMessage({ status: "loading", message: "Loading Model Files" });
      
      const [weightsArrayU8, tokenizerArrayU8, originalConfigU8] = await Promise.all([
        weightsURL instanceof Array ? concatenateArrayBuffers(weightsURL) : fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
        fetchArrayBuffer(configURL),
      ]);

      // Parse the original config and convert it to candle-compatible format
      const originalConfig = JSON.parse(new TextDecoder().decode(originalConfigU8));
      
      // Create candle-compatible config based on Phi-2 parameters
      const candleConfig = {
        "_name_or_path": originalConfig._name_or_path || "microsoft/phi-2",
        "vocab_size": originalConfig.vocab_size || 51200,
        "n_positions": originalConfig.max_position_embeddings || 2048,
        "n_embd": originalConfig.hidden_size || 2560,
        "n_layer": originalConfig.num_hidden_layers || 32,
        "n_inner": originalConfig.intermediate_size || null,
        "n_head": originalConfig.num_attention_heads || 32,
        "rotary_dim": Math.min(32, (originalConfig.hidden_size || 2560) / (originalConfig.num_attention_heads || 32)),
        "activation_function": originalConfig.activation_function || "gelu_new",
        "layer_norm_epsilon": originalConfig.layer_norm_eps || 1e-5,
        "tie_word_embeddings": originalConfig.tie_word_embeddings || false,
        "pad_vocab_size_multiple": originalConfig.pad_vocab_size_multiple || 64
      };

      // Convert the modified config back to bytes
      const configArrayU8 = new TextEncoder().encode(JSON.stringify(candleConfig));

      self.postMessage({ 
        status: "complete", 
        message: "Model files loaded successfully",
        weights: weightsArrayU8,
        tokenizer: tokenizerArrayU8,
        config: configArrayU8
      });
    } catch (e) {
      self.postMessage({ 
        status: "error", 
        message: "Failed to load model files",
        error: e.message || e.toString()
      });
    }
  }
}

self.addEventListener("message", (event) => {
  if (event.data.command === "loadModel") {
    const { weightsURL, tokenizerURL, configURL } = event.data;
    PhiModelLoader.loadModelFiles(weightsURL, tokenizerURL, configURL);
  }
});
