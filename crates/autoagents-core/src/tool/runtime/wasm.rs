use serde_json::Value;
use wasmtime::{Engine, Linker, Module, Store, TypedFunc};

#[derive(Debug, thiserror::Error)]
pub enum WasmRuntimeError {
    #[error("ModuleLoad Error: {0}")]
    ModuleLoad(String),
    #[error("Instantiation Error: {0}")]
    Instantiation(String),
    #[error("MemoryAccess Error: {0}")]
    MemoryAccess(String),
    #[error("Json Error: {0}")]
    JsonError(String),
    #[error("UTF8 Error: {0}")]
    Utf8Error(String),
    #[error("Function Error: {0}")]
    FunctionError(String),
    #[error("Missing Source File")]
    MissingSourceFile,
}

#[derive(Debug, Default)]
pub struct WasmRuntimeConfig {
    pub alloc_fn: String,
    pub execute_fn: String,
    pub free_fn: Option<String>,
}

pub struct WasmRuntime {
    engine: Engine,
    module: Module,
    config: WasmRuntimeConfig,
}

#[derive(Debug, Default)]
pub struct WasmRuntimeBuilder {
    path: Option<String>,
    config: WasmRuntimeConfig,
}

impl WasmRuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn source_file(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn alloc_fn(mut self, name: impl Into<String>) -> Self {
        self.config.alloc_fn = name.into();
        self
    }

    pub fn execute_fn(mut self, name: impl Into<String>) -> Self {
        self.config.execute_fn = name.into();
        self
    }

    pub fn free_fn(mut self, name: Option<String>) -> Self {
        self.config.free_fn = name;
        self
    }

    pub fn build(self) -> Result<WasmRuntime, WasmRuntimeError> {
        let engine = Engine::default();
        let path = self.path.ok_or(WasmRuntimeError::MissingSourceFile)?;
        let module = Module::from_file(&engine, &path)
            .map_err(|e| WasmRuntimeError::ModuleLoad(e.to_string()))?;
        Ok(WasmRuntime {
            engine,
            module,
            config: self.config,
        })
    }
}

impl WasmRuntime {
    pub fn builder() -> WasmRuntimeBuilder {
        WasmRuntimeBuilder::default()
    }

    pub fn run(&self, input: Value) -> Result<Value, WasmRuntimeError> {
        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);

        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmRuntimeError::Instantiation(e.to_string()))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmRuntimeError::MemoryAccess("No exported memory".into()))?;

        let alloc: TypedFunc<i32, i32> = instance
            .get_typed_func(&mut store, &self.config.alloc_fn)
            .map_err(|e| WasmRuntimeError::FunctionError(e.to_string()))?;

        let execute: TypedFunc<(i32, i32), i32> = instance
            .get_typed_func(&mut store, &self.config.execute_fn)
            .map_err(|e| WasmRuntimeError::FunctionError(e.to_string()))?;

        let free: Option<TypedFunc<(i32, i32), ()>> = match &self.config.free_fn {
            Some(free_name) => Some(
                instance
                    .get_typed_func(&mut store, free_name)
                    .map_err(|e| WasmRuntimeError::FunctionError(e.to_string()))?,
            ),
            None => None,
        };

        // Serialize input JSON
        let input_str = serde_json::to_string(&input)
            .map_err(|e| WasmRuntimeError::JsonError(e.to_string()))?;
        let input_bytes = input_str.as_bytes();

        // Allocate memory in wasm
        let ptr = alloc
            .call(&mut store, input_bytes.len() as i32)
            .map_err(|e| WasmRuntimeError::MemoryAccess(e.to_string()))?;

        memory
            .write(&mut store, ptr as usize, input_bytes)
            .map_err(|e| WasmRuntimeError::MemoryAccess(e.to_string()))?;

        // Call execute
        let result_ptr = execute
            .call(&mut store, (ptr, input_bytes.len() as i32))
            .map_err(|e| WasmRuntimeError::MemoryAccess(e.to_string()))?;

        // Read result length
        let mut len_buf = [0u8; 4];
        memory
            .read(&mut store, result_ptr as usize, &mut len_buf)
            .map_err(|e| WasmRuntimeError::MemoryAccess(e.to_string()))?;
        let result_len = i32::from_le_bytes(len_buf) as usize;

        // Read result bytes
        let mut result_bytes = vec![0u8; result_len];
        memory
            .read(&mut store, result_ptr as usize + 4, &mut result_bytes)
            .map_err(|e| WasmRuntimeError::MemoryAccess(e.to_string()))?;

        // Convert to JSON
        let json_str = String::from_utf8(result_bytes)
            .map_err(|e| WasmRuntimeError::Utf8Error(e.to_string()))?;
        let json_value = serde_json::from_str(&json_str)
            .map_err(|e| WasmRuntimeError::JsonError(e.to_string()))?;

        // Free wasm memory after use (if free function exists)
        if let Some(free_func) = free {
            free_func
                .call(&mut store, (result_ptr, (result_len + 4) as i32))
                .map_err(|e| WasmRuntimeError::MemoryAccess(format!("Free failed: {e}")))?;
        }

        Ok(json_value)
    }
}
