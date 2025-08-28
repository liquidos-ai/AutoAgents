export const clientOnly = true;

import {useEffect, useState} from "react";
import type {Route} from "./+types/home";
import init, {TokenStreamer} from "../../../pkg/wasm_agent.js";

export function meta({}: Route.MetaArgs) {
    return [
        {title: "AutoAgents WASM Demo"},
        {name: "description", content: "AutoAgents with Phi Model running in WASM"},
    ];
}

interface ModelFiles {
    weights: Uint8Array;
    tokenizer: Uint8Array;
    config: Uint8Array;
}

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
}

export default function Home() {
    const [wasmInitialized, setWasmInitialized] = useState(false);
    const [modelFiles, setModelFiles] = useState<ModelFiles | null>(null);
    const [tokenStreamer, setTokenStreamer] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState("");
    const [prompt, setPrompt] = useState("");
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [currentStreamingContent, setCurrentStreamingContent] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [selectedModel, setSelectedModel] = useState<string>("phi_1_5_q4k");

    // Model configurations from official candle phi example
    const MODELS = {
        phi_1_5_q4k: {
            name: "Phi 1.5 Q4K (800 MB) - Faster",
            base_url: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/",
            weights: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/model-q4k.gguf",
            tokenizer: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/tokenizer.json",
            config: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/phi-1_5.json",
            quantized: true,
            size: "800 MB"
        },
        phi_2_0_q4k: {
            name: "Phi 2.0 Q4K (1.57 GB) - Better Quality",
            base_url: "https://huggingface.co/radames/phi-2-quantized/resolve/main/",
            weights: [
                "https://huggingface.co/radames/phi-2-quantized/resolve/main/model-v2-q4k.gguf_aa.part",
                "https://huggingface.co/radames/phi-2-quantized/resolve/main/model-v2-q4k.gguf_ab.part",
                "https://huggingface.co/radames/phi-2-quantized/resolve/main/model-v2-q4k.gguf_ac.part"
            ],
            tokenizer: "https://huggingface.co/radames/phi-2-quantized/resolve/main/tokenizer.json",
            config: "https://huggingface.co/radames/phi-2-quantized/resolve/main/config.json",
            quantized: true,
            size: "1.57 GB"
        }
    };

    // @ts-ignore
    const MODEL_URLS = MODELS[selectedModel];

    useEffect(() => {
        initializeWasm();
    }, []);

    const initializeWasm = async () => {
        try {
            await init();
            setWasmInitialized(true);
        } catch (e) {
            setError(`Failed to initialize WASM: ${e}`);
        }
    };

    const loadModelFiles = async () => {
        if (!wasmInitialized) {
            setError("WASM not initialized yet");
            return;
        }

        setLoading(true);
        setError("");
        setLoadingMessage("Loading model files...");

        try {
            // Load files with caching to avoid re-downloading large model
            const cacheName = "phi-mixformer-candle-cache";
            const cache = await caches.open(cacheName);

            // Helper function to fetch with caching
            const fetchWithCache = async (url: string): Promise<Uint8Array> => {
                const cachedResponse = await cache.match(url);
                if (cachedResponse) {
                    setLoadingMessage(`Loading ${url.split('/').pop()} from cache...`);
                    const data = await cachedResponse.arrayBuffer();
                    return new Uint8Array(data);
                }

                setLoadingMessage(`Downloading ${url.split('/').pop()}...`);
                const response = await fetch(url, {cache: "force-cache"});

                // Cache the response
                await cache.put(url, response.clone());

                const arrayBuffer = await response.arrayBuffer();
                return new Uint8Array(arrayBuffer);
            };

            // Helper function to concatenate multiple array buffers (for split models)
            const concatenateArrayBuffers = async (urls: string[]): Promise<Uint8Array> => {
                const arrayBuffers = await Promise.all(urls.map(url => fetchWithCache(url)));
                let totalLength = arrayBuffers.reduce((acc, arrayBuffer) => acc + arrayBuffer.byteLength, 0);
                let concatenatedBuffer = new Uint8Array(totalLength);

                let offset = 0;
                arrayBuffers.forEach(buffer => {
                    concatenatedBuffer.set(buffer, offset);
                    offset += buffer.byteLength;
                });
                return concatenatedBuffer;
            };

            setLoadingMessage("Loading model weights...");
            const weights = Array.isArray(MODEL_URLS.weights)
                ? await concatenateArrayBuffers(MODEL_URLS.weights)
                : await fetchWithCache(MODEL_URLS.weights);

            setLoadingMessage("Loading tokenizer...");
            const tokenizer = await fetchWithCache(MODEL_URLS.tokenizer);

            setLoadingMessage("Loading config...");
            const configCached = await cache.match(MODEL_URLS.config);
            let configText: string;

            if (configCached) {
                setLoadingMessage("Loading config from cache...");
                configText = await configCached.text();
            } else {
                setLoadingMessage("Downloading config...");
                const configResponse = await fetch(MODEL_URLS.config, {cache: "force-cache"});
                await cache.put(MODEL_URLS.config, configResponse.clone());
                configText = await configResponse.text();
            }

            // Apply config transformation based on model type
            const originalConfig = JSON.parse(configText);

            // Phi 1.5 vs Phi 2.0 have different architectures
            const isPhi15 = selectedModel === 'phi_1_5_q4k';

            // Apply config transformation
            let finalConfig;
            if (isPhi15) {
                // For Phi 1.5, use the config as-is since it's already in the correct candle format
                finalConfig = originalConfig;
            } else {
                // For Phi 2.0, transform from HuggingFace to candle format
                finalConfig = {
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
            }

            const config = new TextEncoder().encode(JSON.stringify(finalConfig));

            setModelFiles({weights, tokenizer, config});

            // Initialize streaming models
            try {
                setLoadingMessage("Initializing streaming model...");

                setLoadingMessage("Initializing token streamer...");
                const tokenStreamerInstance = new TokenStreamer(weights, tokenizer, config, MODELS[selectedModel].quantized);
                setTokenStreamer(tokenStreamerInstance);

                setLoadingMessage("Model ready for chat!");
            } catch (e) {
                setError(`Failed to initialize streaming model: ${e}`);
            }

            setLoading(false);
        } catch (e) {
            setError(`Failed to load model files: ${e}`);
            setLoading(false);
        }
    };

    const sendMessage = async () => {
        if (!tokenStreamer || !prompt.trim()) {
            setError("Model not ready or empty prompt");
            return;
        }

        setIsGenerating(true);
        setError("");
        setCurrentStreamingContent("");

        // Add user message to chat
        const userMessage: ChatMessage = {
            role: 'user',
            content: prompt,
            timestamp: new Date()
        };
        setChatMessages(prev => [...prev, userMessage]);

        // Add initial empty assistant message for streaming
        const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: "",
            timestamp: new Date(),
            isStreaming: true
        };
        setChatMessages(prev => [...prev, assistantMessage]);

        // Clear prompt
        const currentPrompt = prompt;
        setPrompt("");

        try {
            let accumulatedContent = "";

            // Create callback function to handle tokens
            const tokenCallback = (token: string) => {
                accumulatedContent += token;
                setCurrentStreamingContent(accumulatedContent);

                // Update the last assistant message with accumulated content
                setChatMessages(prev => {
                    const newMessages = [...prev];
                    const lastMessage = newMessages[newMessages.length - 1];
                    if (lastMessage.role === 'assistant' && lastMessage.isStreaming) {
                        lastMessage.content = accumulatedContent;
                    }
                    return newMessages;
                });
            };

            // Start streaming
            await tokenStreamer.stream_tokens(currentPrompt, tokenCallback);

            // Finalize the message
            setChatMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage.role === 'assistant' && lastMessage.isStreaming) {
                    lastMessage.isStreaming = false;
                }
                return newMessages;
            });

        } catch (e) {
            setError(`Generation failed: ${e}`);
            // Remove the user and assistant messages if generation failed
            setChatMessages(prev => prev.slice(0, -2));
        } finally {
            setIsGenerating(false);
            setCurrentStreamingContent("");
        }
    };

    const clearChat = () => {
        setChatMessages([]);
        setError("");
    };

    return (
        <div className="min-h-screen bg-black text-white">
            <div className="container mx-auto px-4 py-8 h-screen flex flex-col">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-2">
                        AutoAgents WASM Chat
                    </h1>
                    <p className="text-gray-300">
                        Phi Model running with AutoAgents in WebAssembly
                    </p>
                </div>

                {/* Status and Model Management */}
                <div className="bg-gray-800 rounded-lg shadow-xl p-6 mb-6 border border-gray-700">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                            <h3 className="text-lg font-semibold text-white mb-3">Status</h3>
                            <div className="space-y-2">
                                <div className="flex items-center">
                                    <span
                                        className={`w-3 h-3 rounded-full mr-3 ${wasmInitialized ? 'bg-green-500' : 'bg-red-500'}`}></span>
                                    <span
                                        className="text-gray-300">WASM: {wasmInitialized ? 'Ready' : 'Loading...'}</span>
                                </div>
                                <div className="flex items-center">
                                    <span
                                        className={`w-3 h-3 rounded-full mr-3 ${tokenStreamer ? 'bg-green-500' : 'bg-red-500'}`}></span>
                                    <span
                                        className="text-gray-300">Model: {tokenStreamer ? 'Ready' : 'Not Loaded'}</span>
                                </div>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-lg font-semibold text-white mb-3">Model Selection</h3>
                            <select
                                value={selectedModel}
                                onChange={(e) => {
                                    setSelectedModel(e.target.value);
                                    setModelFiles(null);
                                    setError("");
                                }}
                                disabled={loading}
                                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
                            >
                                {Object.entries(MODELS).map(([key, model]) => (
                                    <option key={key} value={key} className="bg-gray-700">
                                        {model.name}
                                    </option>
                                ))}
                            </select>
                            <p className="mt-1 text-sm text-gray-400">
                                Size: {MODELS[selectedModel].size}
                            </p>
                            <button
                                onClick={loadModelFiles}
                                disabled={loading || !wasmInitialized || tokenStreamer !== null}
                                className="mt-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-medium py-2 px-4 rounded-md transition duration-200"
                            >
                                {tokenStreamer ? '✓ Model Loaded' : 'Load Model'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Loading Indicator */}
                {loading && (
                    <div className="bg-blue-900 border border-blue-600 rounded-md p-4 mb-6">
                        <div className="flex items-center">
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-400" fill="none"
                                 viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                        strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor"
                                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span className="text-blue-100">{loadingMessage}</span>
                        </div>
                    </div>
                )}

                {/* Error Display */}
                {error && (
                    <div className="bg-red-900 border border-red-600 rounded-md p-4 mb-6">
                        <div className="flex">
                            <svg className="h-5 w-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd"
                                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                                      clipRule="evenodd"/>
                            </svg>
                            <span className="text-red-100">{error}</span>
                        </div>
                    </div>
                )}

                {/* Chat Interface */}
                <div className="flex-1 flex flex-col bg-gray-800 rounded-lg shadow-xl border border-gray-700">
                    {/* Chat Header */}
                    <div className="flex justify-between items-center p-4 border-b border-gray-700">
                        <h3 className="text-lg font-semibold text-white">Chat with Phi</h3>
                        <button
                            onClick={clearChat}
                            disabled={chatMessages.length === 0}
                            className="bg-gray-600 hover:bg-gray-500 disabled:bg-gray-700 text-white px-3 py-1 rounded-md text-sm transition duration-200"
                        >
                            Clear Chat
                        </button>
                    </div>

                    {/* Chat Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {chatMessages.length === 0 ? (
                            <div className="text-center text-gray-400 mt-12">
                                <p className="text-lg mb-2">Welcome to AutoAgents WASM Chat!</p>
                                <p>Load a model and start chatting with the Phi language model.</p>
                            </div>
                        ) : (
                            chatMessages.map((message, index) => (
                                <div key={index}
                                     className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-3xl p-4 rounded-lg ${
                                        message.role === 'user'
                                            ? 'bg-blue-600 text-white ml-8'
                                            : 'bg-gray-700 text-gray-100 mr-8'
                                    }`}>
                                        <div className="flex items-start space-x-2">
                                            <div
                                                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                                                    message.role === 'user' ? 'bg-blue-700' : 'bg-gray-600'
                                                }`}>
                                                {message.role === 'user' ? 'U' : 'AI'}
                                            </div>
                                            <div className="flex-1">
                                                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                                                    {message.content}
                                                </div>
                                                <div className="text-xs opacity-70 mt-2">
                                                    {message.timestamp.toLocaleTimeString()}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}

                        {/* Generating indicator */}
                        {isGenerating && (
                            <div className="flex justify-start">
                                <div className="bg-gray-700 text-gray-100 max-w-3xl p-4 rounded-lg mr-8">
                                    <div className="flex items-center space-x-2">
                                        <div
                                            className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-sm font-bold">
                                            AI
                                        </div>
                                        <div className="flex items-center space-x-1">
                                            <span className="text-sm">Generating</span>
                                            <div className="flex space-x-1">
                                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"
                                                     style={{animationDelay: '0.2s'}}></div>
                                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"
                                                     style={{animationDelay: '0.4s'}}></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Chat Input */}
                    <div className="p-4 border-t border-gray-700">
                        <div className="flex space-x-3">
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        sendMessage();
                                    }
                                }}
                                placeholder={tokenStreamer ? "Type your message... (Enter to send, Shift+Enter for new line)" : "Load a model first..."}
                                disabled={!tokenStreamer || isGenerating}
                                className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-gray-400 resize-none"
                                rows={2}
                            />
                            <button
                                onClick={sendMessage}
                                disabled={!tokenStreamer || !prompt.trim() || isGenerating}
                                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-3 rounded-md transition duration-200 font-medium"
                            >
                                {isGenerating ? '...' : 'Send'}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
