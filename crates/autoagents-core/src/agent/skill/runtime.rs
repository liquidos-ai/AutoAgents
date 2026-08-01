use crate::agent::skill::prompt::SkillPromptRenderer;
use crate::agent::skill::{
    SkillActivationRequest, SkillConfiguration, SkillError, SkillLifecycle, SkillLifecycleDecision,
    SkillRefreshReport, SkillResourceRequest, SkillSnapshot,
};
use crate::tool::{ToolCallError, ToolRuntime, ToolT, to_llm_tool};
use async_trait::async_trait;
use autoagents_llm::chat::Tool;
use autoagents_protocol::{ActorID, Event, SubmissionId};
use serde::Deserialize;
use serde_json::{Value, json};
#[cfg(not(target_arch = "wasm32"))]
use sha2::{Digest, Sha256};
use std::path::{Component, Path};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

#[cfg(not(target_arch = "wasm32"))]
use crate::agent::skill::resource::SkillResourceDirectory;

#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;

pub use autoagents_protocol::{
    SkillDeactivationReason, SkillEvent, SkillEventKind, SkillOperation,
};

#[derive(Clone)]
pub(crate) struct SkillRuntimeIdentity {
    actor_id: ActorID,
    submission_id: Option<SubmissionId>,
    events: Option<mpsc::Sender<Event>>,
}

impl std::fmt::Debug for SkillRuntimeIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SkillRuntimeIdentity")
            .field("actor_id", &self.actor_id)
            .field("submission_id", &self.submission_id)
            .field("events", &self.events.as_ref().map(|_| "configured"))
            .finish()
    }
}

impl SkillRuntimeIdentity {
    pub(crate) fn new(
        actor_id: ActorID,
        submission_id: Option<SubmissionId>,
        events: Option<mpsc::Sender<Event>>,
    ) -> Self {
        Self {
            actor_id,
            submission_id,
            events,
        }
    }

    fn submission_id(&self) -> Result<SubmissionId, SkillError> {
        self.submission_id.ok_or_else(|| {
            SkillError::PolicyDenied("skill operation requires an active task".to_string())
        })
    }
}

#[derive(Debug)]
struct SkillEventDispatcher {
    identity: SkillRuntimeIdentity,
    session: Arc<crate::agent::skill::SkillSession>,
    lifecycle: Arc<dyn SkillLifecycle>,
}

impl SkillEventDispatcher {
    fn new(
        identity: SkillRuntimeIdentity,
        session: Arc<crate::agent::skill::SkillSession>,
        lifecycle: Arc<dyn SkillLifecycle>,
    ) -> Self {
        Self {
            identity,
            session,
            lifecycle,
        }
    }

    fn event(&self, event: SkillEventKind) -> SkillEvent {
        SkillEvent::new(
            self.identity.actor_id,
            self.identity.submission_id,
            self.session.id(),
            event,
        )
    }

    async fn publish(&self, event: &SkillEvent) {
        let Some(events) = &self.identity.events else {
            return;
        };

        #[cfg(not(target_arch = "wasm32"))]
        let _ = events
            .send(Event::Skill {
                event: event.clone(),
            })
            .await;

        #[cfg(target_arch = "wasm32")]
        {
            let mut events = events.clone();
            let _ = events
                .send(Event::Skill {
                    event: event.clone(),
                })
                .await;
        }
    }

    async fn catalog_changed(&self, event: SkillEvent) {
        self.publish(&event).await;
        self.lifecycle.on_catalog_changed(&event).await;
    }

    async fn activation_requested(&self, event: SkillEvent) -> Result<(), SkillError> {
        self.publish(&event).await;
        match self.lifecycle.on_activation_requested(&event).await {
            SkillLifecycleDecision::Continue => Ok(()),
            SkillLifecycleDecision::Abort => Err(SkillError::PolicyDenied(
                "skill activation aborted by lifecycle hook".to_string(),
            )),
        }
    }

    async fn activated(&self, event: SkillEvent) {
        self.publish(&event).await;
        self.lifecycle.on_activated(&event).await;
    }

    async fn deactivated(&self, event: SkillEvent) {
        self.publish(&event).await;
        self.lifecycle.on_deactivated(&event).await;
    }

    async fn resource_access_requested(&self, event: SkillEvent) -> Result<(), SkillError> {
        self.publish(&event).await;
        match self.lifecycle.on_resource_access_requested(&event).await {
            SkillLifecycleDecision::Continue => Ok(()),
            SkillLifecycleDecision::Abort => Err(SkillError::PolicyDenied(
                "skill resource access aborted by lifecycle hook".to_string(),
            )),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn resource_accessed(&self, event: SkillEvent) {
        self.publish(&event).await;
        self.lifecycle.on_resource_accessed(&event).await;
    }

    async fn operation_failed(&self, event: SkillEvent) {
        self.publish(&event).await;
        self.lifecycle.on_operation_failed(&event).await;
    }
}

#[derive(Debug)]
pub struct SkillRuntime {
    configuration: SkillConfiguration,
    events: SkillEventDispatcher,
}

impl SkillRuntime {
    pub const ACTIVATE_TOOL_NAME: &'static str = "activate_skill";
    pub const DEACTIVATE_TOOL_NAME: &'static str = "deactivate_skill";
    pub const READ_RESOURCE_TOOL_NAME: &'static str = "read_skill_resource";

    pub(crate) fn new(
        configuration: SkillConfiguration,
        identity: SkillRuntimeIdentity,
        lifecycle: Arc<dyn SkillLifecycle>,
    ) -> Arc<Self> {
        let session = configuration.session();
        Arc::new(Self {
            events: SkillEventDispatcher::new(identity, session, lifecycle),
            configuration,
        })
    }

    pub fn tools(self: &Arc<Self>) -> Vec<Box<dyn ToolT>> {
        vec![
            Box::new(ActivateSkillTool {
                runtime: Arc::clone(self),
            }),
            Box::new(DeactivateSkillTool {
                runtime: Arc::clone(self),
            }),
            Box::new(ReadSkillResourceTool {
                runtime: Arc::clone(self),
            }),
        ]
    }

    pub fn serialized_tools(self: &Arc<Self>) -> Vec<Tool> {
        self.tools().iter().map(to_llm_tool).collect()
    }

    pub fn configuration(&self) -> &SkillConfiguration {
        &self.configuration
    }

    pub async fn compose_system_prompt(&self, base_prompt: &str) -> String {
        let snapshot = self.refresh_catalog().await;
        let active = self.configuration.session_ref().active();
        SkillPromptRenderer::render(base_prompt, &snapshot, &active)
    }

    pub async fn refresh_now(&self) -> SkillRefreshReport {
        let report = self.configuration.registry_ref().refresh_now().await;
        self.apply_refresh(&report).await;
        report
    }

    pub(crate) async fn initialize(&self) {
        let report = self.configuration.registry_ref().refresh().await;
        self.apply_refresh(&report).await;
    }

    pub async fn reset_session(&self) {
        let reset = self.configuration.session_ref().reset();
        for skill_name in reset.deactivated() {
            let event = SkillEvent::new(
                self.events.identity.actor_id,
                self.events.identity.submission_id,
                reset.previous_id(),
                SkillEventKind::Deactivated {
                    skill_name: skill_name.clone(),
                    reason: SkillDeactivationReason::SessionReset,
                },
            );
            self.events.deactivated(event).await;
        }
    }

    pub fn validate_agent_tools(tools: &[Box<dyn ToolT>]) -> Result<(), SkillError> {
        for tool in tools {
            if matches!(
                tool.name(),
                Self::ACTIVATE_TOOL_NAME
                    | Self::DEACTIVATE_TOOL_NAME
                    | Self::READ_RESOURCE_TOOL_NAME
            ) {
                return Err(SkillError::ToolNameConflict(tool.name().to_string()));
            }
        }
        Ok(())
    }

    async fn refresh_catalog(&self) -> Arc<SkillSnapshot> {
        let registry = self.configuration.registry_ref();
        let report = registry.refresh().await;
        self.apply_refresh(&report).await;
        registry.snapshot()
    }

    async fn apply_refresh(&self, report: &SkillRefreshReport) {
        report.log_diagnostics();

        if report.changed() {
            let event = self.events.event(SkillEventKind::CatalogChanged {
                generation: report.generation(),
                added: report.added().to_vec(),
                updated: report.updated().to_vec(),
                removed: report.removed().to_vec(),
            });
            self.events.catalog_changed(event).await;
        }

        for diagnostic in report.diagnostics() {
            let event = self.events.event(SkillEventKind::OperationFailed {
                operation: SkillOperation::RefreshCatalog,
                skill_name: None,
                message: format!("{}: {}", diagnostic.source().as_str(), diagnostic.message()),
            });
            self.events.operation_failed(event).await;
        }

        let snapshot = self.configuration.registry_ref().snapshot();
        let reconciliation = self.configuration.session_ref().reconcile(&snapshot);
        for skill_name in reconciliation.updated() {
            let event = self.events.event(SkillEventKind::Deactivated {
                skill_name: skill_name.clone(),
                reason: SkillDeactivationReason::Updated,
            });
            self.events.deactivated(event).await;
        }
        for skill_name in reconciliation.removed() {
            let event = self.events.event(SkillEventKind::Deactivated {
                skill_name: skill_name.clone(),
                reason: SkillDeactivationReason::Removed,
            });
            self.events.deactivated(event).await;
        }
    }

    async fn activate(&self, name: String) -> Result<Value, SkillError> {
        let snapshot = self.refresh_catalog().await;
        let skill = snapshot
            .get(&name)
            .ok_or_else(|| SkillError::UnknownSkill(name.clone()))?;
        let requested = self.events.event(SkillEventKind::ActivationRequested {
            skill_name: name.clone(),
        });
        self.events.activation_requested(requested).await?;

        let request = SkillActivationRequest::new(
            self.events.identity.actor_id,
            self.events.identity.submission_id()?,
            self.configuration.session_ref().id(),
            Arc::clone(&skill),
        );
        self.configuration
            .policy_ref()
            .authorize_activation(&request)
            .await?;

        let newly_activated = self
            .configuration
            .session_ref()
            .activate(name.clone(), skill.revision());
        let activated = self.events.event(SkillEventKind::Activated {
            skill_name: name,
            newly_activated,
        });
        self.events.activated(activated).await;
        let resources = skill
            .resources()
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>();
        let allowed_tools = skill
            .metadata()
            .allowed_tools()
            .iter()
            .map(|selector| selector.expression())
            .collect::<Vec<_>>();

        Ok(json!({
            "activated": skill.metadata().name(),
            "newly_activated": newly_activated,
            "resources": resources,
            "allowed_tools": allowed_tools,
            "next_step": "The skill instructions are now present in the system prompt. Follow them before taking other actions."
        }))
    }

    async fn deactivate(&self, name: String) -> Result<Value, SkillError> {
        if !self.configuration.session_ref().deactivate(&name) {
            return Err(SkillError::InactiveSkill(name));
        }
        let event = self.events.event(SkillEventKind::Deactivated {
            skill_name: name.clone(),
            reason: SkillDeactivationReason::Requested,
        });
        self.events.deactivated(event).await;
        Ok(json!({"deactivated": name}))
    }

    async fn read_resource(
        &self,
        skill_name: String,
        resource: String,
    ) -> Result<Value, SkillError> {
        let relative = SkillResourcePath::parse(&resource)?;
        let snapshot = self.refresh_catalog().await;
        let skill = snapshot
            .get(&skill_name)
            .ok_or_else(|| SkillError::UnknownSkill(skill_name.clone()))?;
        if !self.configuration.session_ref().is_active(&skill_name) {
            return Err(SkillError::InactiveSkill(skill_name));
        }
        if !skill
            .resources()
            .iter()
            .any(|path| path == relative.as_path())
        {
            return Err(SkillError::InvalidResourcePath(resource));
        }

        let requested = self.events.event(SkillEventKind::ResourceAccessRequested {
            skill_name: skill_name.clone(),
            path: resource.clone(),
        });
        self.events.resource_access_requested(requested).await?;
        let request = SkillResourceRequest::new(
            self.events.identity.actor_id,
            self.events.identity.submission_id()?,
            self.configuration.session_ref().id(),
            Arc::clone(&skill),
            relative.as_path().to_path_buf(),
        );
        self.configuration
            .policy_ref()
            .authorize_resource_read(&request)
            .await?;

        let relative_path = relative.as_path().to_path_buf();

        #[cfg(target_arch = "wasm32")]
        {
            let _ = relative_path;
            return Err(SkillError::source_unavailable(
                skill_name,
                "filesystem skill resources are unavailable on wasm32",
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let reader = SkillResourceReader {
                skill_name: skill_name.clone(),
                directory: skill.resource_directory().ok_or_else(|| {
                    SkillError::source_unavailable(
                        &skill_name,
                        "skill source did not provide a confined resource directory",
                    )
                })?,
                display_directory: skill.directory().to_path_buf(),
                relative: relative_path.clone(),
                expected_digest: skill.resource_digest(&relative_path).ok_or_else(|| {
                    SkillError::source_unavailable(
                        &skill_name,
                        "skill source did not provide a resource revision",
                    )
                })?,
                max_bytes: self.configuration.max_resource_bytes(),
            };
            let bytes = tokio::task::spawn_blocking(move || reader.read())
                .await
                .map_err(|error| SkillError::SourceWorker(error.to_string()))??;

            let byte_count = bytes.len();
            let content = String::from_utf8(bytes)
                .map_err(|_| SkillError::ResourceNotUtf8(skill.directory().join(&relative_path)))?;
            let accessed = self.events.event(SkillEventKind::ResourceAccessed {
                skill_name: skill_name.clone(),
                path: resource.clone(),
                bytes: byte_count,
            });
            self.events.resource_accessed(accessed).await;
            Ok(json!({
                "skill": skill_name,
                "path": resource,
                "content": content
            }))
        }
    }

    async fn report_failure(
        &self,
        operation: SkillOperation,
        skill_name: Option<String>,
        error: &SkillError,
    ) {
        let event = self.events.event(SkillEventKind::OperationFailed {
            operation,
            skill_name,
            message: error.to_string(),
        });
        self.events.operation_failed(event).await;
    }
}

#[derive(Debug)]
struct SkillResourcePath {
    path: std::path::PathBuf,
}

impl SkillResourcePath {
    fn parse(path: &str) -> Result<Self, SkillError> {
        let path = Path::new(path);
        if path.as_os_str().is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(SkillError::InvalidResourcePath(path.display().to_string()));
        }
        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    fn as_path(&self) -> &Path {
        &self.path
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct SkillResourceReader {
    skill_name: String,
    directory: SkillResourceDirectory,
    display_directory: PathBuf,
    relative: PathBuf,
    expected_digest: [u8; 32],
    max_bytes: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl SkillResourceReader {
    fn read(self) -> Result<Vec<u8>, SkillError> {
        let requested = self.display_directory.join(&self.relative);
        let file = self.directory.open_file(&self.relative).map_err(|error| {
            SkillError::source_unavailable(
                &self.skill_name,
                format!("cannot open resource '{}': {error}", requested.display()),
            )
        })?;
        let metadata = file.metadata().map_err(|error| {
            SkillError::source_unavailable(
                &self.skill_name,
                format!("cannot inspect resource '{}': {error}", requested.display()),
            )
        })?;
        if !metadata.is_file() {
            return Err(SkillError::InvalidResourcePath(
                self.relative.display().to_string(),
            ));
        }
        if metadata.len() > self.max_bytes as u64 {
            return Err(SkillError::ResourceTooLarge {
                path: requested,
                limit: self.max_bytes,
            });
        }

        let mut bytes = Vec::with_capacity((metadata.len() as usize).min(self.max_bytes));
        file.take(self.max_bytes.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| {
                SkillError::source_unavailable(
                    &self.skill_name,
                    format!("cannot read resource '{}': {error}", requested.display()),
                )
            })?;
        if bytes.len() > self.max_bytes {
            return Err(SkillError::ResourceTooLarge {
                path: requested,
                limit: self.max_bytes,
            });
        }
        let observed_digest: [u8; 32] = Sha256::digest(&bytes).into();
        if observed_digest != self.expected_digest {
            return Err(SkillError::ResourceChanged(requested));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Deserialize)]
struct ActivateSkillArgs {
    name: String,
}

#[derive(Debug)]
struct ActivateSkillTool {
    runtime: Arc<SkillRuntime>,
}

impl ToolT for ActivateSkillTool {
    fn name(&self) -> &str {
        SkillRuntime::ACTIVATE_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Activate one available Agent Skill by its exact name. Call this before using the skill's instructions or resources."
    }

    fn args_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact skill name from the available skills catalog"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }
}

#[async_trait]
impl ToolRuntime for ActivateSkillTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let args: ActivateSkillArgs = serde_json::from_value(args)?;
        match self.runtime.activate(args.name.clone()).await {
            Ok(value) => Ok(value),
            Err(error) => {
                self.runtime
                    .report_failure(SkillOperation::Activate, Some(args.name), &error)
                    .await;
                Err(ToolCallError::RuntimeError(Box::new(error)))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct DeactivateSkillArgs {
    name: String,
}

#[derive(Debug)]
struct DeactivateSkillTool {
    runtime: Arc<SkillRuntime>,
}

impl ToolT for DeactivateSkillTool {
    fn name(&self) -> &str {
        SkillRuntime::DEACTIVATE_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Deactivate one Agent Skill for the current conversation when its instructions no longer apply."
    }

    fn args_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact name of an active skill"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }
}

#[async_trait]
impl ToolRuntime for DeactivateSkillTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let args: DeactivateSkillArgs = serde_json::from_value(args)?;
        match self.runtime.deactivate(args.name.clone()).await {
            Ok(value) => Ok(value),
            Err(error) => {
                self.runtime
                    .report_failure(SkillOperation::Deactivate, Some(args.name), &error)
                    .await;
                Err(ToolCallError::RuntimeError(Box::new(error)))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadSkillResourceArgs {
    skill: String,
    path: String,
}

#[derive(Debug)]
struct ReadSkillResourceTool {
    runtime: Arc<SkillRuntime>,
}

impl ToolT for ReadSkillResourceTool {
    fn name(&self) -> &str {
        SkillRuntime::READ_RESOURCE_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Read one UTF-8 resource file from an activated Agent Skill. The path must exactly match a resource listed for that skill."
    }

    fn args_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Exact name of an activated skill"
                },
                "path": {
                    "type": "string",
                    "description": "Relative resource path listed by the activated skill"
                }
            },
            "required": ["skill", "path"],
            "additionalProperties": false
        })
    }
}

#[async_trait]
impl ToolRuntime for ReadSkillResourceTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let args: ReadSkillResourceArgs = serde_json::from_value(args)?;
        match self
            .runtime
            .read_resource(args.skill.clone(), args.path)
            .await
        {
            Ok(value) => Ok(value),
            Err(error) => {
                self.runtime
                    .report_failure(SkillOperation::ReadResource, Some(args.skill), &error)
                    .await;
                Err(ToolCallError::RuntimeError(Box::new(error)))
            }
        }
    }
}
