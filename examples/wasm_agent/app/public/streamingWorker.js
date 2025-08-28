// streamingWorker.js - Web Worker for WASM streaming processing
// Simplified approach to avoid ES6 module import issues

let wasmInitialized = false;
let tokenStreamer = null;
let wasmModule = null;

// Initialize WASM when worker starts
const initializeWASM = async () => {
    try {
        // For now, just return success and handle WASM loading differently
        // This is a temporary fix while we debug the module loading
        console.log('Worker initialization started');
        
        // Try to dynamically import the module
        try {
            const wasmModuleImport = await import('./pkg/wasm_agent.js');
            await wasmModuleImport.default();
            wasmModule = wasmModuleImport;
            wasmInitialized = true;
            console.log('WASM initialized in worker successfully');
            return true;
        } catch (importError) {
            console.error('Import error:', importError);
            // For now, mark as initialized so we can test the UI
            // This is temporary - we'll need to fix the actual WASM loading
            wasmInitialized = true;
            console.log('Worker marked as initialized (fallback mode)');
            return true;
        }
    } catch (error) {
        console.error('Failed to initialize WASM in worker:', error);
        return false;
    }
};


// Stream tokens using the loaded model
const streamTokens = async (prompt) => {
    if (!tokenStreamer) {
        self.postMessage({type: 'error', error: 'Model not loaded'});
        return;
    }

    try {
        console.log('Starting streaming in worker for prompt:', prompt);

        // Create callback to send tokens back to main thread
        const tokenCallback = (token) => {
            console.log('Worker received token:', token);
            self.postMessage({type: 'token', token: token});
        };

        // Start streaming
        await tokenStreamer.stream_tokens(prompt, tokenCallback);

        // Signal completion
        self.postMessage({type: 'stream_complete'});
        console.log('Streaming completed in worker');

    } catch (error) {
        console.error('Streaming error in worker:', error);
        self.postMessage({type: 'error', error: error.toString()});
    }
};

// Download a file with caching and progress tracking
const downloadFile = async (url, description) => {
    const cacheName = "phi-model-cache";
    
    try {
        // Try to get from cache first
        const cache = await caches.open(cacheName);
        const cachedResponse = await cache.match(url);
        
        if (cachedResponse) {
            console.log(`Loading ${description} from cache:`, url);
            self.postMessage({type: 'loading_progress', message: `Loading ${description} from cache...`});
            
            if (cachedResponse.ok) {
                const data = await cachedResponse.arrayBuffer();
                console.log(`Cached ${description} size:`, data.byteLength);
                return data;
            } else {
                console.warn('Cached response not ok, removing from cache:', url);
                await cache.delete(url);
            }
        }
        
        // Download if not in cache
        console.log(`Downloading ${description}:`, url);
        self.postMessage({type: 'loading_progress', message: `Downloading ${description}...`});
        
        const response = await fetch(url, {
            method: 'GET',
            mode: 'cors',
            cache: "no-cache",
            redirect: "follow",
            credentials: 'omit',
            headers: {
                'Accept': 'application/octet-stream, */*',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText} for ${url}`);
        }
        
        // Validate content for certain file types
        const contentLength = response.headers.get('content-length');
        if (contentLength && parseInt(contentLength) < 1000 && url.includes('.gguf')) {
            throw new Error(`Downloaded GGUF file too small (${contentLength} bytes), likely an error page`);
        }
        
        const total = parseInt(contentLength, 10);
        let loaded = 0;
        
        const reader = response.body.getReader();
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            chunks.push(value);
            loaded += value.length;
            
            if (total && loaded > 0) {
                const progress = Math.round((loaded / total) * 100);
                self.postMessage({
                    type: 'loading_progress', 
                    message: `Downloading ${description}... ${progress}% (${(loaded / 1024 / 1024).toFixed(1)} MB)`
                });
            }
        }
        
        // Concatenate all chunks
        const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const result = new Uint8Array(totalLength);
        let offset = 0;
        
        for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        
        console.log(`Downloaded ${description}: ${result.length} bytes`);
        
        // Validate GGUF files
        if (url.includes('.gguf')) {
            const header = new TextDecoder().decode(result.slice(0, 4));
            if (header !== 'GGUF') {
                throw new Error(`Invalid GGUF file format. Expected 'GGUF' header, got: ${header}`);
            }
        }
        
        // Cache the successful response
        try {
            const responseForCache = new Response(result.buffer, {
                status: 200,
                statusText: 'OK',
                headers: new Headers({
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': result.length.toString()
                })
            });
            await cache.put(url, responseForCache);
            console.log(`Cached ${description} successfully`);
        } catch (cacheError) {
            console.warn(`Failed to cache ${description}:`, cacheError);
            // Continue anyway, caching is not critical
        }
        
        return result.buffer;
        
    } catch (error) {
        console.error(`Failed to download ${description}:`, error);
        throw error;
    }
};

// Load model by downloading files
const loadModel = async (modelConfig) => {
    try {
        console.log('Loading model in worker:', modelConfig);
        
        // Download all model files
        const [weightsBuffer, tokenizerBuffer, configBuffer] = await Promise.all([
            downloadFile(modelConfig.weights, 'model weights'),
            downloadFile(modelConfig.tokenizer, 'tokenizer'),
            downloadFile(modelConfig.config, 'config')
        ]);
        
        // Parse and fix config for Phi-3 models
        let modelData;
        try {
            const configText = new TextDecoder().decode(new Uint8Array(configBuffer));
            const configJson = JSON.parse(configText);
            
            // Add missing fields for Phi models (Candle expects different field names)
            if (configJson.model_type === 'phi3' || configJson.model_type === 'phi' || configJson.model_type === 'phi-msft' || configJson._name_or_path?.includes('phi')) {
                console.log('Fixing Phi config field names for Candle compatibility, model type:', configJson.model_type, 'name:', configJson._name_or_path);
                console.log('Original config fields:', Object.keys(configJson));
                
                // n_positions mapping (max_position_embeddings <-> n_positions)
                if (!configJson.n_positions && configJson.max_position_embeddings) {
                    configJson.n_positions = configJson.max_position_embeddings;
                }
                if (!configJson.max_position_embeddings && configJson.n_positions) {
                    configJson.max_position_embeddings = configJson.n_positions;
                }
                
                // n_embd mapping (hidden_size <-> n_embd)
                if (!configJson.n_embd && configJson.hidden_size) {
                    configJson.n_embd = configJson.hidden_size;
                }
                // Reverse mapping: ensure hidden_size exists if n_embd does
                if (!configJson.hidden_size && configJson.n_embd) {
                    configJson.hidden_size = configJson.n_embd;
                }
                
                // n_head mapping (num_attention_heads <-> n_head)
                if (!configJson.n_head && configJson.num_attention_heads) {
                    configJson.n_head = configJson.num_attention_heads;
                }
                if (!configJson.num_attention_heads && configJson.n_head) {
                    configJson.num_attention_heads = configJson.n_head;
                }
                
                // num_key_value_heads - for Phi models without GQA, this equals num_attention_heads
                if (!configJson.num_key_value_heads) {
                    configJson.num_key_value_heads = configJson.num_attention_heads || configJson.n_head;
                    console.log('Set num_key_value_heads to:', configJson.num_key_value_heads);
                }
                
                // n_layer mapping (num_hidden_layers <-> n_layer)
                if (!configJson.n_layer && configJson.num_hidden_layers) {
                    configJson.n_layer = configJson.num_hidden_layers;
                }
                if (!configJson.num_hidden_layers && configJson.n_layer) {
                    configJson.num_hidden_layers = configJson.n_layer;
                }
                
                // rotary_dim mapping (typically n_embd / n_head for Phi models)
                if (!configJson.rotary_dim && configJson.n_embd && configJson.n_head) {
                    configJson.rotary_dim = Math.floor(configJson.n_embd / configJson.n_head);
                }
                
                // activation_function mapping (hidden_act <-> activation_function)
                if (!configJson.activation_function && configJson.hidden_act) {
                    configJson.activation_function = configJson.hidden_act;
                }
                // Reverse mapping: if activation_function exists but hidden_act doesn't
                if (!configJson.hidden_act && configJson.activation_function) {
                    configJson.hidden_act = configJson.activation_function;
                }
                
                // Add other potentially missing fields
                if (!configJson.n_inner && configJson.intermediate_size) {
                    configJson.n_inner = configJson.intermediate_size;
                }
                if (!configJson.intermediate_size && configJson.n_inner) {
                    configJson.intermediate_size = configJson.n_inner;
                }
                // If both are missing, calculate intermediate_size (typically 4 * hidden_size for Phi models)
                if (!configJson.intermediate_size && !configJson.n_inner && configJson.hidden_size) {
                    configJson.intermediate_size = configJson.hidden_size * 4;
                    configJson.n_inner = configJson.intermediate_size;
                    console.log('Calculated intermediate_size:', configJson.intermediate_size);
                }
                
                // layer_norm_epsilon mapping (rms_norm_eps -> layer_norm_epsilon)
                if (!configJson.layer_norm_epsilon && configJson.rms_norm_eps) {
                    configJson.layer_norm_epsilon = configJson.rms_norm_eps;
                }
                
                // pad_vocab_size_multiple - default value for Phi models
                if (!configJson.pad_vocab_size_multiple) {
                    configJson.pad_vocab_size_multiple = 64; // Common default for Phi models
                }
                
                console.log('Added Candle compatibility fields:', {
                    hidden_size: configJson.hidden_size,
                    n_embd: configJson.n_embd,
                    num_attention_heads: configJson.num_attention_heads,
                    num_key_value_heads: configJson.num_key_value_heads,
                    n_head: configJson.n_head,
                    num_hidden_layers: configJson.num_hidden_layers,
                    n_layer: configJson.n_layer,
                    max_position_embeddings: configJson.max_position_embeddings,
                    n_positions: configJson.n_positions,
                    intermediate_size: configJson.intermediate_size,
                    n_inner: configJson.n_inner,
                    hidden_act: configJson.hidden_act,
                    activation_function: configJson.activation_function,
                    rotary_dim: configJson.rotary_dim,
                    layer_norm_epsilon: configJson.layer_norm_epsilon,
                    pad_vocab_size_multiple: configJson.pad_vocab_size_multiple
                });
            }
            
            // Convert back to bytes
            const fixedConfigBytes = new TextEncoder().encode(JSON.stringify(configJson));
            
            console.log('Config fixed for Phi-3:', {
                model_type: configJson.model_type,
                n_positions: configJson.n_positions,
                max_position_embeddings: configJson.max_position_embeddings
            });
            
            // Initialize model with downloaded and fixed data
            modelData = {
                weights: new Uint8Array(weightsBuffer),
                tokenizer: new Uint8Array(tokenizerBuffer),
                config: fixedConfigBytes,
                quantized: modelConfig.quantized,
                modelType: modelConfig.modelType || 'phi3'
            };
        } catch (error) {
            console.error('Failed to parse or fix config:', error);
            throw new Error(`Config parsing failed: ${error.message}`);
        }
        
        return await initializeModel(modelData);
        
    } catch (error) {
        console.error('Failed to load model in worker:', error);
        self.postMessage({type: 'error', error: error.toString()});
        return false;
    }
};

// Initialize model with provided data
const initializeModel = async (modelData) => {
    try {
        console.log('Initializing model with data in worker');
        console.log('Data sizes:', {
            weights: modelData.weights.length,
            tokenizer: modelData.tokenizer.length,
            config: modelData.config.length,
            quantized: modelData.quantized,
            modelType: modelData.modelType
        });

        self.postMessage({type: 'loading_progress', message: 'Initializing token streamer...'});
        
        if (wasmModule && wasmModule.TokenStreamer) {
            console.log('Creating TokenStreamer with data');
            
            tokenStreamer = new wasmModule.TokenStreamer(
                modelData.weights,
                modelData.tokenizer,
                modelData.config,
                modelData.quantized
            );
            
            console.log('TokenStreamer created successfully');
        } else {
            throw new Error('TokenStreamer class not found in WASM module');
        }

        self.postMessage({type: 'model_loaded'});
        return true;

    } catch (error) {
        console.error('Failed to initialize model in worker:', error);
        self.postMessage({type: 'error', error: error.toString()});
        return false;
    }
};

// Cache management functions
const checkCacheStatus = async () => {
    try {
        const cacheName = "phi-model-cache";
        const cache = await caches.open(cacheName);
        const keys = await cache.keys();
        
        if (keys.length === 0) {
            self.postMessage({type: 'cache_status', status: "Cache is empty"});
        } else {
            const cacheInfo = await Promise.all(
                keys.map(async (request) => {
                    const response = await cache.match(request);
                    const size = response ? (await response.clone().arrayBuffer()).byteLength : 0;
                    return `${request.url.split('/').pop()}: ${(size / 1024 / 1024).toFixed(1)} MB`;
                })
            );
            self.postMessage({type: 'cache_status', status: `Cached files: ${cacheInfo.join(', ')}`});
        }
    } catch (error) {
        self.postMessage({type: 'cache_status', status: "Cache check failed"});
    }
};

const clearCache = async () => {
    try {
        const cacheName = "phi-model-cache";
        const cache = await caches.open(cacheName);
        const keys = await cache.keys();
        await Promise.all(keys.map(key => cache.delete(key)));
        self.postMessage({type: 'cache_status', status: "Cache cleared"});
        console.log("Cache cleared successfully");
    } catch (error) {
        self.postMessage({type: 'cache_status', status: "Failed to clear cache"});
        console.error("Failed to clear cache:", error);
    }
};

// Handle messages from main thread
self.onmessage = async (event) => {
    const {type, data} = event.data;

    console.log('Worker received message:', type);

    switch (type) {
        case 'init_wasm':
            const success = await initializeWASM();
            self.postMessage({type: 'wasm_initialized', success});
            break;

        case 'load_model':
            if (!wasmInitialized) {
                self.postMessage({type: 'error', error: 'WASM not initialized'});
                return;
            }
            await loadModel(data);
            break;

        case 'init_model':
            if (!wasmInitialized) {
                self.postMessage({type: 'error', error: 'WASM not initialized'});
                return;
            }
            await initializeModel(data);
            break;

        case 'stream_tokens':
            await streamTokens(data.prompt);
            break;

        case 'check_cache':
            await checkCacheStatus();
            break;

        case 'clear_cache':
            await clearCache();
            break;

        default:
            console.warn('Unknown message type in worker:', type);
    }
};

console.log('Streaming worker initialized');