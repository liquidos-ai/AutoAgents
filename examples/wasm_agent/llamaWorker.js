import init, { LLamaChatWrapper } from "./pkg/wasm_agent.js";

// Cache API storage using service worker pattern in web worker
const CACHE_NAME = 'llama-autoagents-cache-v1';

// Check storage availability
function checkStorageSupport() {
  const hasCacheAPI = typeof caches !== 'undefined';
  console.log('Storage support check:', {
    cacheAPI: hasCacheAPI,
    userAgent: navigator.userAgent
  });
  return { hasCacheAPI };
}

async function getCachedModel(url) {
  try {
    if (typeof caches === 'undefined') {
      console.log('Cache API not available');
      return null;
    }

    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(url);

    if (cachedResponse) {
      const data = await cachedResponse.arrayBuffer();
      const sizeInMB = (data.byteLength / (1024 * 1024)).toFixed(2);
      console.log(`✅ Found cached model for ${url}, size: ${sizeInMB} MB`);
      return new Uint8Array(data);
    } else {
      console.log(`No cached model found for ${url}`);
      return null;
    }
  } catch (error) {
    console.warn("Cache API read failed:", error);
    return null;
  }
}

async function setCachedModel(url, data) {
  try {
    if (typeof caches === 'undefined') {
      console.warn('Cache API not available for storing');
      return;
    }

    const sizeInMB = (data.length / (1024 * 1024)).toFixed(2);
    console.log(`Caching ${url}, size: ${sizeInMB} MB`);

    const cache = await caches.open(CACHE_NAME);

    // Create a Response object from the data
    const response = new Response(data, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': data.length.toString(),
        'Cache-Control': 'max-age=31536000' // Cache for 1 year
      }
    });

    await cache.put(url, response);
    console.log(`✅ Successfully cached ${url} (${sizeInMB} MB)`);
  } catch (error) {
    console.error("Cache API write failed:", {
      error: error.message,
      name: error.name,
      url: url
    });
    // Don't throw, just continue without caching
  }
}

// Debug function to list cached models
async function listCachedModels() {
  try {
    if (typeof caches === 'undefined') return [];

    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();

    const models = [];
    for (const request of requests) {
      const response = await cache.match(request);
      const contentLength = response.headers.get('Content-Length');
      models.push({
        url: request.url,
        size: contentLength ? parseInt(contentLength) : 'unknown',
        sizeFormatted: contentLength ? `${(parseInt(contentLength) / (1024 * 1024)).toFixed(2)} MB` : 'unknown'
      });
    }

    console.log('Cached models:', models);
    return models;
  } catch (error) {
    console.warn('Error listing cached models:', error);
    return [];
  }
}

async function fetchArrayBuffer(url) {
  const storage = checkStorageSupport();

  // Debug: List what's currently cached
  if (storage.hasCacheAPI) {
    await listCachedModels();
  }

  // Try to get from Cache API first
  if (storage.hasCacheAPI) {
    console.log(`Checking cache for ${url}...`);
    const cached = await getCachedModel(url);
    if (cached && cached.length > 0) {
      console.log(`✅ Loading ${url} from Cache API (${cached.length} bytes)`);
      self.postMessage({ status: "loading", message: "Loading from cache..." });
      return cached;
    }
  }

  console.log(`📥 Downloading ${url}${storage.hasCacheAPI ? ' (will cache for next time)' : ' (no cache available)'}`);
  self.postMessage({ status: "loading", message: "Downloading model files..." });

  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }

  const data = new Uint8Array(await res.arrayBuffer());
  console.log(`Downloaded ${url}: ${data.length} bytes`);

  // Store in Cache API for next time if available
  if (storage.hasCacheAPI) {
    try {
      await setCachedModel(url, data);
      console.log(`✅ Cached ${url} in Cache API for future use`);
    } catch (cacheError) {
      console.warn(`Failed to cache ${url}:`, cacheError);
    }
  } else {
    console.warn('⚠️ No Cache API available - model will be downloaded each time');
  }

  return data;
}

class LlamaModel {
  static instance = null;

  static async getInstance(weightsURL, tokenizerURL) {
    if (!this.instance) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Llama Model" });
      const [weightsArrayU8, tokenizerArrayU8] = await Promise.all([
        fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
      ]);

      self.postMessage({ status: "loading", message: "Initializing Llama Agent" });
      this.instance = await LLamaChatWrapper.create(weightsArrayU8, tokenizerArrayU8);
    }
    return this.instance;
  }
}

let controller = null;
self.addEventListener("message", (event) => {
  if (event.data.command === "start") {
    controller = new AbortController();
    generate(event.data);
  } else if (event.data.command === "abort") {
    controller.abort();
  }
});

async function generate(data) {
  const { weightsURL, tokenizerURL, prompt } = data;

  try {
    self.postMessage({ status: "loading", message: "Starting Llama" });
    const model = await LlamaModel.getInstance(weightsURL, tokenizerURL);

    self.postMessage({ status: "loading", message: "Generating response" });

    let sentence = "";
    let startTime = performance.now();
    let tokensCount = 0;

    // Create a callback function to handle streaming tokens
    const tokenCallback = (token) => {
      if (controller && controller.signal.aborted) {
        self.postMessage({
          status: "aborted",
          message: "Aborted",
          output: sentence,
        });
        return;
      }

      tokensCount++;
      const tokensSec = (tokensCount / (performance.now() - startTime)) * 1000;
      sentence += token;

      self.postMessage({
        status: "generating",
        message: "Generating token",
        token: token,
        sentence: sentence,
        totalTime: performance.now() - startTime,
        tokensSec,
        prompt: prompt,
      });
    };

    // Use the streaming method
    await model.get_response_stream(prompt, tokenCallback);

    self.postMessage({
      status: "complete",
      message: "complete",
      output: sentence,
    });

  } catch (e) {
    console.error("Generation error:", e);
    self.postMessage({ error: e.toString() });
  }
}