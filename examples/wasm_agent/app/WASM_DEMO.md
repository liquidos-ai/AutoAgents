# AutoAgents WASM Demo

This React application demonstrates AutoAgents running with the Phi model in WebAssembly.

## Features

- **WASM Integration**: AutoAgents framework compiled to WebAssembly
- **Phi Model**: Microsoft Phi-2 model running in the browser
- **Web Worker**: Model file loading handled in a separate thread
- **React UI**: Modern, responsive interface with Tailwind CSS

## How it Works

1. **WASM Module**: The Rust AutoAgents code is compiled to WASM and can be imported as a JavaScript module
2. **Phi Provider**: A custom LLM provider that implements the AutoAgents `LLMProvider` and `ChatProvider` traits
3. **Model Loading**: Web Worker loads model files (weights, tokenizer, config) from HuggingFace
4. **Agent Execution**: Math agent with Addition tool processes user prompts using the Phi model

## UI Components

### Status Panel
- Shows WASM module initialization status
- Shows model file loading status

### Model Management
- Button to load model files from HuggingFace
- Uses Web Worker for non-blocking downloads
- Caches files using browser cache API

### Chat Interface
- Text area for user input
- Run Agent button to process requests
- Displays agent responses

## Model Files

The demo uses these model files:
- **Weights**: `https://huggingface.co/lmz/candle-quantized-phi/resolve/main/model-q4k.gguf`
- **Tokenizer**: `https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json`  
- **Config**: `https://huggingface.co/microsoft/phi-2/resolve/main/config.json`

## Building and Running

1. Build the WASM module:
   ```bash
   cd /path/to/wasm_agent
   ./build-lib.sh
   ```

2. Install dependencies and run the React app:
   ```bash
   cd app
   npm install
   npm run dev
   ```

3. Open `http://localhost:5173` in your browser

## Technical Details

- **Framework**: React with React Router v7
- **Styling**: Tailwind CSS
- **WASM Binding**: wasm-bindgen
- **Model**: Quantized Phi-2 via Candle-Transformers
- **Execution**: Single-threaded agent execution in WASM

## Usage Flow

1. Page loads and initializes WASM module
2. Click "Load Model Files" to download and cache model data
3. Enter a prompt (e.g., "What is 2 + 2?")
4. Click "Run Agent" to process with the Math Agent
5. View the agent's response with reasoning

The Math Agent can use the Addition tool for mathematical operations and provide explanations for its reasoning.