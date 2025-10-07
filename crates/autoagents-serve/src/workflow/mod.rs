pub mod direct;
pub mod llm_factory;
pub mod parallel;
pub mod routing;
pub mod sequential;
pub mod types;

pub use direct::DirectWorkflow;
pub use parallel::ParallelWorkflow;
pub use routing::RoutingWorkflow;
pub use sequential::SequentialWorkflow;
pub use types::{Workflow, WorkflowOutput};

// Type alias for model cache to simplify signatures
pub type ModelCache = std::sync::Arc<
    tokio::sync::RwLock<
        std::collections::HashMap<String, std::sync::Arc<dyn autoagents::llm::LLMProvider>>,
    >,
>;
