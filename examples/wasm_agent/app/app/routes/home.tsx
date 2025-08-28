export const clientOnly = true;

import {useEffect, useState} from "react";
import type {Route} from "./+types/home";

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
    const [loadingProgress, setLoadingProgress] = useState(0);
    const [prompt, setPrompt] = useState("");
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [currentStreamingContent, setCurrentStreamingContent] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [selectedModel, setSelectedModel] = useState<string>("phi_1_5_q4k");
    const [worker, setWorker] = useState<Worker | null>(null);
    const [cacheStatus, setCacheStatus] = useState<string>("");

    // Model configurations - Working models only
    const MODELS = {
        phi_1_5_q4k: {
            name: "Phi 1.5 Q4K (800 MB) - Reliable",
            base_url: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/",
            weights: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/model-q4k.gguf",
            tokenizer: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/tokenizer.json",
            config: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/phi-1_5.json",
            quantized: true,
            size: "800 MB"
        },
        phi_3_mini_4k_gguf: {
            name: "Phi-3 Mini 4K Instruct Q2_K (1.6 GB) - QuantFactory",
            base_url: "https://huggingface.co/QuantFactory/Phi-3-mini-4k-instruct-GGUF/resolve/main/",
            weights: "https://huggingface.co/QuantFactory/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct.Q2_K.gguf",
            tokenizer: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.json",
            config: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/config.json",
            quantized: true,
            size: "1.6 GB",
            modelType: "phi3"
        }
    };

    // @ts-ignore
    const MODEL_URLS = MODELS[selectedModel];

    useEffect(() => {
        initializeWorker();
        return () => {
            // Cleanup worker on unmount
            if (worker) {
                worker.terminate();
            }
        };
    }, []);

    // Check cache status when worker is ready
    useEffect(() => {
        if (worker && wasmInitialized) {
            checkCacheStatus();
        }
    }, [worker, wasmInitialized]);

    const initializeWorker = async () => {
        try {
            console.log('Initializing worker...');
            const newWorker = new Worker('/streamingWorker.js');

            // Set up worker message handler
            newWorker.onmessage = (event) => {
                const {type, data, message, token, error, success} = event.data;
                console.log('Received from worker:', type, data);

                switch (type) {
                    case 'wasm_initialized':
                        setWasmInitialized(success);
                        if (!success) {
                            setError('Failed to initialize WASM in worker');
                        }
                        break;

                    case 'loading_progress':
                        setLoadingMessage(message);
                        // Extract percentage from message if it contains one
                        const percentMatch = message.match(/(\d+)%/);
                        if (percentMatch) {
                            const percent = parseInt(percentMatch[1]);
                            setLoadingProgress(percent);
                        } else {
                            // For non-percentage messages, show some progress based on stage
                            if (message.includes('Downloading')) {
                                setLoadingProgress(10); // Starting download
                            } else if (message.includes('cache')) {
                                setLoadingProgress(90); // Loading from cache is fast
                            } else if (message.includes('Initializing')) {
                                setLoadingProgress(95); // Almost done
                            }
                        }
                        break;

                    case 'model_loaded':
                        setTokenStreamer(true); // Just a flag to indicate model is ready
                        setLoading(false);
                        setLoadingMessage("Model ready for chat!");
                        setLoadingProgress(100);
                        break;

                    case 'token':
                        handleTokenReceived(token);
                        break;

                    case 'stream_complete':
                        handleStreamComplete();
                        break;

                    case 'cache_status':
                        setCacheStatus(event.data.status);
                        break;

                    case 'error':
                        setError(error);
                        setLoading(false);
                        setIsGenerating(false);
                        setLoadingProgress(0);
                        break;
                }
            };

            newWorker.onerror = (error) => {
                console.error('Worker error:', error);
                setError('Worker initialization failed');
            };

            setWorker(newWorker);

            // Initialize WASM in worker
            newWorker.postMessage({type: 'init_wasm'});

        } catch (e) {
            console.error('Failed to initialize worker:', e);
            setError(`Failed to initialize worker: ${e}`);
        }
    };

    // Handler for tokens received from worker
    const handleTokenReceived = (token: string) => {
        console.log('UI received token:', token);

        // Skip empty tokens to avoid duplication
        if (!token || token.trim() === '') {
            return;
        }

        setChatMessages(prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
                // Ensure we don't add the same token twice
                if (!lastMessage.content.endsWith(token)) {
                    lastMessage.content += token;
                }
            }
            return newMessages;
        });
    };

    // Handler for stream completion
    const handleStreamComplete = () => {
        console.log('Stream completed');
        setChatMessages(prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
                lastMessage.isStreaming = false;
            }
            return newMessages;
        });
        setIsGenerating(false);
        setCurrentStreamingContent("");
    };

    // Cache management functions
    const checkCacheStatus = () => {
        if (worker) {
            worker.postMessage({type: 'check_cache'});
        }
    };

    const clearCache = () => {
        if (worker) {
            worker.postMessage({type: 'clear_cache'});
        }
    };

    const loadModelFiles = async () => {
        if (!wasmInitialized || !worker) {
            setError("Worker not initialized yet");
            return;
        }

        setLoading(true);
        setError("");
        setLoadingMessage("Starting model download...");
        setLoadingProgress(0);

        try {
            const modelConfig = MODELS[selectedModel];
            
            // Send model configuration to worker for download and initialization
            worker.postMessage({
                type: 'load_model',
                data: modelConfig
            });

        } catch (e) {
            setError(`Failed to load model: ${e}`);
            setLoading(false);
            setLoadingMessage("");
            setLoadingProgress(0);
        }
    };

    // Cache management functions removed - now handled in Web Worker

    const sendMessage = async () => {
        if (!tokenStreamer || !prompt.trim() || !worker) {
            setError("Model not ready, empty prompt, or worker not available");
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

        // Clear prompt and start streaming in worker
        const currentPrompt = prompt;
        setPrompt("");

        try {
            // Send streaming request to worker
            worker.postMessage({
                type: 'stream_tokens',
                data: {prompt: currentPrompt}
            });
        } catch (e) {
            setError(`Failed to start generation: ${e}`);
            // Remove the user and assistant messages if generation failed to start
            setChatMessages(prev => prev.slice(0, -2));
            setIsGenerating(false);
            setCurrentStreamingContent("");
        }
    };

    const clearChat = () => {
        setChatMessages([]);
        setError("");
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-zinc-900 via-slate-900 to-zinc-900">
            {/* Background Pattern */}
            <div
                className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%23ffffff%22%20fill-opacity%3D%220.05%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%221%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-50"></div>

            <div className="relative min-h-screen flex flex-col">
                {/* Navigation Header */}
                <nav className="backdrop-blur-xl bg-zinc-900/80 border-b border-zinc-700/60">
                    <div className="w-full mx-auto px-6 py-4">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-4">
                                <img src="/logo.png" alt="AutoAgents Logo" className="w-8 h-8 rounded-lg"/>
                                <div>
                                    <h1 className="text-xl font-bold text-white">
                                        AutoAgents
                                    </h1>
                                    <p className="text-xs text-gray-300">
                                        Agentic AI Platform
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center space-x-3">
                                <div className="flex items-center space-x-2 text-xs text-gray-300">
                                    <span
                                        className={`w-2 h-2 rounded-full ${wasmInitialized ? 'bg-green-400' : 'bg-red-400'}`}></span>
                                    <span>Runtime</span>
                                </div>
                                <div className="flex items-center space-x-2 text-xs text-gray-300">
                                    <span
                                        className={`w-2 h-2 rounded-full ${tokenStreamer ? 'bg-green-400' : 'bg-amber-400'}`}></span>
                                    <span>Agent</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </nav>

                {/* Main Content Area */}
                <div className="flex-1 mx-auto px-6 py-6 flex gap-6 h-full w-full">
                    {/* Left Sidebar - Agent Configuration */}
                    <div className="w-80 flex-shrink-0 h-full">
                        <div
                            className="backdrop-blur-xl bg-zinc-800/60 rounded-2xl border border-zinc-600/50 p-6 h-full">
                            <div className="mb-6">
                                <div className="mb-3">
                                    <div
                                        className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mb-3">
                                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                                            <path
                                                d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
                                        </svg>
                                    </div>
                                    <div>
                                        <h2 className="text-lg font-semibold text-white mb-1">Agent Control</h2>
                                        <p className="text-gray-300 text-xs mb-3">Configure and manage your AI agent</p>
                                    </div>
                                </div>
                                <div className="flex flex-col space-y-1 h-full">
                                    <span className="text-xs text-gray-300 font-medium">Status</span>
                                    <span className={`px-3 py-1 rounded-lg text-xs font-medium w-fit ${
                                        tokenStreamer ? 'bg-green-500/30 text-green-300 border border-green-500/40' : 'bg-amber-500/30 text-amber-300 border border-amber-500/40'
                                    }`}>
                                        {tokenStreamer ? 'Ready' : wasmInitialized ? 'Configuring' : 'Initializing'}
                                    </span>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-white mb-2">Model Selection</label>
                                    <select
                                        value={selectedModel}
                                        onChange={(e) => {
                                            setSelectedModel(e.target.value);
                                            setModelFiles(null);
                                            setError("");
                                        }}
                                        disabled={loading}
                                        className="w-full px-3 py-2 bg-zinc-700/80 border border-zinc-500/50 rounded-lg text-white placeholder-zinc-300 focus:outline-none focus:ring-2 focus:ring-purple-400/60 focus:border-transparent text-sm backdrop-blur-xl"
                                    >
                                        {Object.entries(MODELS).map(([key, model]) => (
                                            <option key={key} value={key} className="bg-gray-800 text-white">
                                                {model.name}
                                            </option>
                                        ))}
                                    </select>
                                    <p className="mt-1 text-xs text-gray-300">
                                        Size: {MODELS[selectedModel].size}
                                    </p>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-white mb-2">Cache Status</label>
                                    <div className="p-3 bg-zinc-700/50 border border-zinc-500/30 rounded-lg">
                                        <p className="text-xs text-gray-300 mb-2">{cacheStatus || "Checking cache..."}</p>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={checkCacheStatus}
                                                className="px-2 py-1 bg-blue-600/80 text-white text-xs rounded hover:bg-blue-600 transition-colors"
                                            >
                                                Refresh
                                            </button>
                                            <button
                                                onClick={clearCache}
                                                className="px-2 py-1 bg-red-600/80 text-white text-xs rounded hover:bg-red-600 transition-colors"
                                            >
                                                Clear Cache
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <button
                                    onClick={loadModelFiles}
                                    disabled={loading || !wasmInitialized || tokenStreamer !== null}
                                    className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:from-gray-600 disabled:to-gray-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 text-sm flex items-center justify-center space-x-2"
                                >
                                    {loading ? (
                                        <>
                                            <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10"
                                                        stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor"
                                                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            <span>Loading...</span>
                                        </>
                                    ) : tokenStreamer ? (
                                        <>
                                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                                <path fillRule="evenodd"
                                                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                                      clipRule="evenodd"/>
                                            </svg>
                                            <span>Agent Ready</span>
                                        </>
                                    ) : (
                                        <span>Initialize Agent</span>
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Right Side - Chat Interface */}
                    <div className="flex-1 flex flex-col">
                        {/* Loading Progress */}
                        {loading && (
                            <div className="mb-4">
                                <div
                                    className="backdrop-blur-xl bg-purple-500/25 rounded-2xl border border-purple-400/50 p-4">
                                    <div className="flex items-center space-x-3">
                                        <div className="flex-shrink-0">
                                            <div className="w-8 h-8 relative">
                                                <div
                                                    className="w-8 h-8 rounded-full border-2 border-purple-500/30"></div>
                                                <div
                                                    className="w-8 h-8 rounded-full border-2 border-purple-500 border-t-transparent animate-spin absolute top-0"></div>
                                            </div>
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex justify-between items-center mb-1">
                                                <p className="text-purple-200 font-medium text-sm">Initializing Agent</p>
                                                <span className="text-purple-300 text-xs font-mono">{loadingProgress}%</span>
                                            </div>
                                            <p className="text-purple-300 text-xs">{loadingMessage}</p>
                                        </div>
                                    </div>
                                    {/* Progress bar */}
                                    <div className="mt-3 w-full bg-purple-900/30 rounded-full h-1">
                                        <div
                                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-1 rounded-full transition-all duration-300 ease-out"
                                            style={{width: `${loadingProgress}%`}}></div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Error Display */}
                        {error && (
                            <div className="mb-4">
                                <div
                                    className="backdrop-blur-xl bg-red-500/25 rounded-2xl border border-red-400/50 p-4">
                                    <div className="flex items-center space-x-3">
                                        <div
                                            className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center">
                                            <svg className="h-4 w-4 text-red-400" fill="currentColor"
                                                 viewBox="0 0 20 20">
                                                <path fillRule="evenodd"
                                                      d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                                      clipRule="evenodd"/>
                                            </svg>
                                        </div>
                                        <div className="flex-1">
                                            <p className="text-red-200 font-medium text-sm">Agent Error</p>
                                            <p className="text-red-300 text-xs">{error}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Main Chat Interface */}
                        <div className="flex-1 flex flex-col">
                            <div
                                className="backdrop-blur-xl bg-zinc-800/50 rounded-2xl border border-zinc-600/50 h-full flex flex-col">
                                {/* Chat Header */}
                                <div className="flex justify-between items-center p-6 border-b border-zinc-600/50">
                                    <div className="flex items-center space-x-3">
                                        <div
                                            className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                                            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                                                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                            </svg>
                                        </div>
                                        <div>
                                            <h3 className="text-lg font-semibold text-white">
                                                {MODELS[selectedModel].name.split(' ')[0]} Agent
                                            </h3>
                                            <p className="text-xs text-gray-300">
                                                {isGenerating ? 'Processing...' : tokenStreamer ? 'Ready to assist' : 'Select and initialize a model'}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <button
                                            onClick={clearChat}
                                            disabled={chatMessages.length === 0}
                                            className="px-3 py-1.5 text-xs font-medium text-gray-300 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            Clear History
                                        </button>
                                    </div>
                                </div>

                                {/* Chat Messages */}
                                <div className="flex-1 overflow-y-auto">
                                    {chatMessages.length === 0 ? (
                                        <div
                                            className="flex flex-col items-center justify-center h-full text-center p-12">
                                            <div
                                                className="w-20 h-20 mb-6 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center backdrop-blur-xl border border-white/10">
                                                <svg className="w-10 h-10 text-purple-400" fill="none"
                                                     stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                                                          d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                                                </svg>
                                            </div>
                                            <h3 className="text-xl font-semibold text-white mb-3">Ready to Assist</h3>
                                            <p className="text-gray-400 max-w-md">
                                                Your AI agent is ready to help. Start a conversation to explore its
                                                capabilities.
                                            </p>
                                            {!tokenStreamer && (
                                                <div
                                                    className="mt-6 px-4 py-2 bg-amber-500/20 text-amber-300 rounded-lg text-sm">
                                                    Initialize an agent model to begin
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="space-y-6 p-6">
                                            {chatMessages.map((message, index) => (
                                                <div key={index}
                                                     className={`group ${message.role === 'assistant' ? 'ml-0' : 'mr-0'}`}>
                                                    <div
                                                        className={`flex items-start space-x-4 ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                                                        <div
                                                            className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                                                                message.role === 'user'
                                                                    ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white'
                                                                    : 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white'
                                                            }`}>
                                                            {message.role === 'user' ? (
                                                                <svg className="w-5 h-5" fill="currentColor"
                                                                     viewBox="0 0 20 20">
                                                                    <path fillRule="evenodd"
                                                                          d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
                                                                          clipRule="evenodd"/>
                                                                </svg>
                                                            ) : (
                                                                <svg className="w-5 h-5" fill="currentColor"
                                                                     viewBox="0 0 20 20">
                                                                    <path
                                                                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                                                </svg>
                                                            )}
                                                        </div>
                                                        <div
                                                            className={`flex-1 min-w-0 ${message.role === 'user' ? 'text-right' : ''}`}>
                                                            <div className={`inline-block max-w-3xl p-4 rounded-2xl ${
                                                                message.role === 'user'
                                                                    ? 'bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 ml-auto'
                                                                    : 'bg-white/5 border border-white/10'
                                                            } backdrop-blur-xl`}>
                                                                <div
                                                                    className="whitespace-pre-wrap text-sm leading-relaxed text-white">
                                                                    {message.content}
                                                                    {message.isStreaming && (
                                                                        <span
                                                                            className="inline-block w-2 h-4 bg-blue-400 ml-1 animate-pulse"></span>
                                                                    )}
                                                                </div>
                                                            </div>
                                                            <div
                                                                className={`text-xs text-gray-400 mt-2 ${message.role === 'user' ? 'text-right' : ''}`}>
                                                                {message.timestamp.toLocaleTimeString()}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* Generating indicator */}
                                    {isGenerating && chatMessages.length > 0 && !chatMessages[chatMessages.length - 1].isStreaming && (
                                        <div className="p-6">
                                            <div className="flex items-start space-x-4">
                                                <div
                                                    className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white flex items-center justify-center">
                                                    <svg className="w-5 h-5 animate-spin" fill="none"
                                                         viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10"
                                                                stroke="currentColor" strokeWidth="4"></circle>
                                                        <path className="opacity-75" fill="currentColor"
                                                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                    </svg>
                                                </div>
                                                <div className="flex-1">
                                                    <div
                                                        className="inline-block p-4 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-xl">
                                                        <div className="flex items-center space-x-3">
                                                            <span
                                                                className="text-sm text-gray-300">Agent is processing</span>
                                                            <div className="flex space-x-1">
                                                                <div
                                                                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                                                                <div
                                                                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                                                                    style={{animationDelay: '0.1s'}}></div>
                                                                <div
                                                                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                                                                    style={{animationDelay: '0.2s'}}></div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Chat Input */}
                                <div className="p-6 border-t border-zinc-600/50 items-center">
                                    <div className="relative">
                                        <div className="flex items-end space-x-4">
                                            <div className="flex-1 relative">
                                        <textarea
                                            value={prompt}
                                            onChange={(e) => setPrompt(e.target.value)}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter' && !e.shiftKey) {
                                                    e.preventDefault();
                                                    sendMessage();
                                                }
                                            }}
                                            placeholder={tokenStreamer ? "Ask your agent anything..." : "Initialize an agent model first"}
                                            disabled={!tokenStreamer || isGenerating}
                                            className="w-full px-6 py-4 bg-zinc-700/80 border border-zinc-500/50 rounded-2xl text-white placeholder-zinc-300 focus:outline-none focus:ring-2 focus:ring-purple-400/60 focus:border-transparent resize-none text-sm backdrop-blur-xl min-h-[56px] max-h-32"
                                            rows={1}
                                        />
                                            </div>
                                            <button
                                                onClick={sendMessage}
                                                disabled={!tokenStreamer || !prompt.trim() || isGenerating}
                                                className="flex-shrink-0 w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-xl transition-all duration-200 flex items-center justify-center group"
                                            >
                                                {isGenerating ? (
                                                    <svg className="w-5 h-5 animate-spin" fill="none"
                                                         viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10"
                                                                stroke="currentColor" strokeWidth="4"></circle>
                                                        <path className="opacity-75" fill="currentColor"
                                                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                    </svg>
                                                ) : (
                                                    <svg
                                                        className="w-5 h-5 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform"
                                                        fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round"
                                                              strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                                                    </svg>
                                                )}
                                            </button>
                                        </div>
                                        <div className="flex items-center justify-between mt-3 text-xs text-gray-300">
                                            <span>Press Enter to send, Shift+Enter for new line</span>
                                            <span>Powered by AutoAgents + WebAssembly</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
