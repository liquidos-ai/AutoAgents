use super::SkillRuntimeIdentity;
use super::{
    DirectorySkillSource, SilentSkillLifecycle, SkillActivationRequest, SkillConfiguration,
    SkillDiscoveryLimits, SkillError, SkillLifecycle, SkillLifecycleDecision, SkillPolicy,
    SkillRefreshStrategy, SkillRegistry, SkillResourceRequest, SkillRuntime, SkillSession,
    SkillSource, SkillSourceId, SkillSourceSnapshot, TrustedSkillPolicy,
};
use crate::agent::memory::SlidingWindowMemory;
use crate::agent::prebuilt::executor::ReActAgent;
use crate::agent::{AgentBuilder, AgentDeriveT, AgentHooks, DirectAgent};
use crate::tests::{MockAgentImpl, MockLLMProvider, StaticChatResponse};
use crate::tool::{ToolCallError, ToolRuntime, ToolT};
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::ModelsProvider;
use autoagents_llm::{FunctionCall, LLMProvider, ToolCall};
use autoagents_protocol::{Event, SkillEvent, SkillEventKind};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tempfile::TempDir;
use uuid::Uuid;

const TEST_REFRESH: SkillRefreshStrategy = SkillRefreshStrategy::Poll {
    interval: std::time::Duration::ZERO,
};

struct SkillFixture {
    _temporary: TempDir,
    root: PathBuf,
}

#[derive(Clone, Debug)]
struct StringSkillsAgent;

impl AgentDeriveT for StringSkillsAgent {
    type Output = String;

    fn description(&self) -> &str {
        "Use matching skills."
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        None
    }

    fn name(&self) -> &str {
        "string_skills_agent"
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        Vec::new()
    }
}

impl AgentHooks for StringSkillsAgent {}

#[derive(Debug, Default)]
struct ScriptedSkillsProvider {
    turn: AtomicUsize,
    active_prompt_seen: AtomicBool,
}

#[derive(Debug)]
struct CountingSkillSource {
    id: SkillSourceId,
    calls: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct HighVolumeDiagnosticSkillSource {
    id: SkillSourceId,
    diagnostic_count: usize,
}

#[derive(Debug)]
struct ReservedNameTool;

#[derive(Debug)]
struct DenyActivationPolicy;

#[derive(Debug, Default)]
struct PausingActivationPolicy {
    entered: tokio::sync::Notify,
    proceed: tokio::sync::Notify,
}

#[async_trait]
impl SkillPolicy for DenyActivationPolicy {
    async fn authorize_activation(
        &self,
        request: &SkillActivationRequest,
    ) -> Result<(), SkillError> {
        Err(SkillError::PolicyDenied(format!(
            "{} is not approved",
            request.skill().metadata().name()
        )))
    }

    async fn authorize_resource_read(
        &self,
        _request: &SkillResourceRequest,
    ) -> Result<(), SkillError> {
        Ok(())
    }
}

#[async_trait]
impl SkillPolicy for PausingActivationPolicy {
    async fn authorize_activation(
        &self,
        _request: &SkillActivationRequest,
    ) -> Result<(), SkillError> {
        self.entered.notify_one();
        self.proceed.notified().await;
        Ok(())
    }

    async fn authorize_resource_read(
        &self,
        _request: &SkillResourceRequest,
    ) -> Result<(), SkillError> {
        Ok(())
    }
}

#[derive(Debug, Default)]
struct RecordingSkillLifecycle {
    events: Mutex<Vec<SkillEventKind>>,
}

impl RecordingSkillLifecycle {
    fn record(&self, event: &SkillEvent) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event.event.clone());
    }

    fn observed(&self) -> Vec<SkillEventKind> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl SkillLifecycle for RecordingSkillLifecycle {
    async fn on_catalog_changed(&self, event: &SkillEvent) {
        self.record(event);
    }

    async fn on_activation_requested(&self, event: &SkillEvent) -> SkillLifecycleDecision {
        self.record(event);
        SkillLifecycleDecision::Continue
    }

    async fn on_activated(&self, event: &SkillEvent) {
        self.record(event);
    }

    async fn on_deactivated(&self, event: &SkillEvent) {
        self.record(event);
    }

    async fn on_resource_access_requested(&self, event: &SkillEvent) -> SkillLifecycleDecision {
        self.record(event);
        SkillLifecycleDecision::Continue
    }

    async fn on_resource_accessed(&self, event: &SkillEvent) {
        self.record(event);
    }

    async fn on_operation_failed(&self, event: &SkillEvent) {
        self.record(event);
    }
}

impl ToolT for ReservedNameTool {
    fn name(&self) -> &str {
        SkillRuntime::ACTIVATE_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Conflicts with the Agent Skills activation tool."
    }

    fn args_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
}

#[async_trait]
impl ToolRuntime for ReservedNameTool {
    async fn execute(&self, _args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        Ok(serde_json::Value::Null)
    }
}

#[async_trait]
impl SkillSource for CountingSkillSource {
    fn id(&self) -> &SkillSourceId {
        &self.id
    }

    fn precedence(&self) -> u16 {
        0
    }

    async fn discover(&self) -> Result<SkillSourceSnapshot, SkillError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        Ok(SkillSourceSnapshot::default())
    }
}

#[async_trait]
impl SkillSource for HighVolumeDiagnosticSkillSource {
    fn id(&self) -> &SkillSourceId {
        &self.id
    }

    fn precedence(&self) -> u16 {
        0
    }

    async fn discover(&self) -> Result<SkillSourceSnapshot, SkillError> {
        let diagnostics = (0..self.diagnostic_count)
            .map(|index| format!("initial diagnostic {index}"))
            .collect();
        Ok(SkillSourceSnapshot::new(Vec::new(), diagnostics))
    }
}

impl ScriptedSkillsProvider {
    fn active_prompt_seen(&self) -> bool {
        self.active_prompt_seen.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ChatProvider for ScriptedSkillsProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None, json_schema).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let turn = self.turn.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            return Ok(Box::new(StaticChatResponse {
                text: None,
                tool_calls: Some(vec![ToolCall {
                    id: "activate-1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: SkillRuntime::ACTIVATE_TOOL_NAME.to_string(),
                        arguments: r#"{"name":"release-notes"}"#.to_string(),
                    },
                }]),
                usage: None,
                thinking: None,
            }));
        }

        let active_prompt_seen = messages
            .first()
            .is_some_and(|message| message.content.contains("Use active voice."));
        self.active_prompt_seen
            .store(active_prompt_seen, Ordering::SeqCst);
        Ok(Box::new(StaticChatResponse {
            text: Some("skill applied".to_string()),
            tool_calls: None,
            usage: None,
            thinking: None,
        }))
    }
}

#[async_trait]
impl CompletionProvider for ScriptedSkillsProvider {
    async fn complete(
        &self,
        _request: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "unused".to_string(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for ScriptedSkillsProvider {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl ModelsProvider for ScriptedSkillsProvider {}

impl LLMProvider for ScriptedSkillsProvider {}

impl SkillFixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().expect("temporary directory should be created");
        let root = temporary.path().join("skills");
        fs::create_dir_all(&root).expect("skills directory should be created");
        Self {
            _temporary: temporary,
            root,
        }
    }

    fn root(&self) -> &Path {
        &self.root
    }

    fn write_skill(&self, name: &str, description: &str, instructions: &str) {
        let directory = self.root.join(name);
        fs::create_dir_all(directory.join("references"))
            .expect("skill directories should be created");
        fs::write(
            directory.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n"),
        )
        .expect("SKILL.md should be written");
        fs::write(
            directory.join("references/details.md"),
            format!("Reference for {name}"),
        )
        .expect("skill resource should be written");
    }

    fn remove_skill(&self, name: &str) {
        fs::remove_dir_all(self.root.join(name)).expect("skill directory should be removed");
    }

    fn overwrite_document(&self, name: &str, document: &str) {
        fs::write(self.root.join(name).join("SKILL.md"), document)
            .expect("SKILL.md should be overwritten");
    }

    fn configuration(&self, registry: Arc<SkillRegistry>) -> SkillConfiguration {
        SkillConfiguration::new(
            registry,
            Arc::new(SkillSession::new()),
            Arc::new(TrustedSkillPolicy),
        )
    }

    fn runtime(&self, registry: Arc<SkillRegistry>) -> Arc<SkillRuntime> {
        SkillRuntime::new(
            self.configuration(registry),
            SkillRuntimeIdentity::new(Uuid::new_v4(), Some(Uuid::new_v4()), None),
            Arc::new(SilentSkillLifecycle),
        )
    }

    fn runtime_with_resource_limit(
        &self,
        registry: Arc<SkillRegistry>,
        max_resource_bytes: usize,
    ) -> Arc<SkillRuntime> {
        let configuration = self
            .configuration(registry)
            .with_max_resource_bytes(max_resource_bytes);
        SkillRuntime::new(
            configuration,
            SkillRuntimeIdentity::new(Uuid::new_v4(), Some(Uuid::new_v4()), None),
            Arc::new(SilentSkillLifecycle),
        )
    }
}

#[tokio::test]
async fn agent_build_discovers_skills_at_boot() {
    let fixture = SkillFixture::new();
    fixture.write_skill(
        "release-notes",
        "Write product release notes.",
        "Use active voice.",
    );
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));

    let handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MockAgentImpl::new(
        "skills",
        "Use skills",
    )))
    .llm(Arc::new(MockLLMProvider))
    .memory(Box::new(SlidingWindowMemory::new(8)))
    .skills(fixture.configuration(Arc::clone(&registry)))
    .build()
    .await
    .expect("agent should build");

    assert_eq!(registry.snapshot().len(), 1);
    assert!(registry.get("release-notes").is_some());
    assert_eq!(handle.agent.tools().len(), 0);
}

#[tokio::test]
async fn direct_agent_build_does_not_block_when_initial_skill_events_exceed_channel_capacity() {
    use futures_util::StreamExt;

    let diagnostic_count = crate::agent::constants::DEFAULT_CHANNEL_BUFFER + 1;
    let registry = Arc::new(SkillRegistry::new(vec![Arc::new(
        HighVolumeDiagnosticSkillSource {
            id: SkillSourceId::new("high-volume-diagnostics"),
            diagnostic_count,
        },
    )]));
    let configuration = SkillConfiguration::new(
        Arc::clone(&registry),
        Arc::new(SkillSession::new()),
        Arc::new(TrustedSkillPolicy),
    );
    let build = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MockAgentImpl::new(
        "skills",
        "Use skills",
    )))
    .llm(Arc::new(MockLLMProvider))
    .skills(configuration)
    .build();

    let mut handle = tokio::time::timeout(std::time::Duration::from_secs(5), build)
        .await
        .expect("initial skill events must not block direct-agent construction")
        .expect("agent should build");

    assert!(registry.snapshot().is_empty());
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        handle
            .rx
            .by_ref()
            .take(diagnostic_count)
            .collect::<Vec<_>>(),
    )
    .await
    .expect("every initial diagnostic event should be delivered");
    assert_eq!(events.len(), diagnostic_count);
    let mut messages = std::collections::BTreeSet::new();
    for (index, event) in events.into_iter().enumerate() {
        let message = match event {
            Event::Skill {
                event:
                    SkillEvent {
                        event: SkillEventKind::OperationFailed { message, .. },
                        ..
                    },
            } => message,
            other => panic!("expected diagnostic event {index}, got {other:?}"),
        };
        assert!(
            messages.insert(message.clone()),
            "diagnostic event was delivered more than once: {message}"
        );
    }
    for index in 0..diagnostic_count {
        assert!(
            messages.contains(&format!(
                "high-volume-diagnostics: initial diagnostic {index}"
            )),
            "initial diagnostic {index} was not delivered"
        );
    }

    tokio::time::timeout(std::time::Duration::from_millis(100), handle.rx.next())
        .await
        .expect_err("startup replay should contain no duplicate events");
    drop(handle);
}

#[tokio::test]
async fn unsupported_executor_is_rejected_when_skills_are_configured() {
    let fixture = SkillFixture::new();
    fixture.write_skill("release-notes", "Write release notes.", "Be concise.");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));

    let error = match AgentBuilder::<_, DirectAgent>::new(MockAgentImpl::new("basic", "basic"))
        .llm(Arc::new(MockLLMProvider))
        .skills(fixture.configuration(registry))
        .build()
        .await
    {
        Ok(_) => panic!("unsupported executor should fail"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("does not support Agent Skills"));
}

#[tokio::test]
async fn react_agent_activates_skill_and_receives_instructions_next_turn() {
    let fixture = SkillFixture::new();
    fixture.write_skill(
        "release-notes",
        "Write product release notes.",
        "Use active voice.",
    );
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let provider = Arc::new(ScriptedSkillsProvider::default());
    let llm: Arc<dyn LLMProvider> = provider.clone();
    let handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(StringSkillsAgent))
        .llm(llm)
        .memory(Box::new(SlidingWindowMemory::new(8)))
        .skills(fixture.configuration(registry))
        .build()
        .await
        .expect("agent should build");

    let result = handle
        .agent
        .run(crate::agent::task::Task::new("Write a release note"))
        .await
        .expect("agent should complete");

    assert_eq!(result, "skill applied");
    assert!(provider.active_prompt_seen());

    provider.active_prompt_seen.store(false, Ordering::SeqCst);
    handle
        .agent
        .run(crate::agent::task::Task::new("Revise the release note"))
        .await
        .expect("second conversation turn should complete");
    assert!(
        provider.active_prompt_seen(),
        "activation must survive a second agent.run call"
    );

    handle.agent.reset_skill_session().await;
    provider.active_prompt_seen.store(false, Ordering::SeqCst);
    handle
        .agent
        .run(crate::agent::task::Task::new("Start another conversation"))
        .await
        .expect("turn after reset should complete");
    assert!(!provider.active_prompt_seen());
}

#[tokio::test]
async fn runtime_refreshes_added_updated_and_removed_skills() {
    let fixture = SkillFixture::new();
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let runtime = fixture.runtime(Arc::clone(&registry));

    let initial = runtime.compose_system_prompt("base prompt").await;
    assert_eq!(initial, "base prompt");

    fixture.write_skill(
        "release-notes",
        "Write product release notes.",
        "Use active voice.",
    );
    let added = runtime.compose_system_prompt("base prompt").await;
    assert!(added.contains("`release-notes`: Write product release notes."));

    let activate = runtime
        .tools()
        .into_iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist");
    activate
        .execute(json!({"name": "release-notes"}))
        .await
        .expect("skill should activate");
    let active = runtime.compose_system_prompt("base prompt").await;
    assert!(active.contains("## Active skill: `release-notes`"));
    assert!(active.contains("Use active voice."));

    fixture.write_skill(
        "release-notes",
        "Write product release notes.",
        "Use short declarative sentences.",
    );
    let updated = runtime.compose_system_prompt("base prompt").await;
    assert!(!updated.contains("## Active skill: `release-notes`"));
    assert!(!updated.contains("Use short declarative sentences."));
    assert!(!updated.contains("Use active voice."));

    activate
        .execute(json!({"name": "release-notes"}))
        .await
        .expect("updated skill should require reactivation");
    let reactivated = runtime.compose_system_prompt("base prompt").await;
    assert!(reactivated.contains("Use short declarative sentences."));

    fixture.write_skill("incident-review", "Review incidents.", "Start with impact.");
    let second = runtime.compose_system_prompt("base prompt").await;
    assert!(second.contains("`incident-review`: Review incidents."));

    fixture.remove_skill("release-notes");
    let removed = runtime.compose_system_prompt("base prompt").await;
    assert!(!removed.contains("release-notes"));
    assert!(removed.contains("incident-review"));
}

#[tokio::test]
async fn approval_for_an_old_revision_cannot_activate_a_new_revision() {
    let fixture = SkillFixture::new();
    fixture.write_skill("review", "Review changes.", "Use revision A.");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let policy = Arc::new(PausingActivationPolicy::default());
    let configuration = SkillConfiguration::new(
        Arc::clone(&registry),
        Arc::new(SkillSession::new()),
        policy.clone(),
    );
    let runtime = SkillRuntime::new(
        configuration,
        SkillRuntimeIdentity::new(Uuid::new_v4(), Some(Uuid::new_v4()), None),
        Arc::new(SilentSkillLifecycle),
    );
    runtime.compose_system_prompt("base").await;
    let activation = runtime
        .tools()
        .into_iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool");
    let worker = tokio::spawn(async move { activation.execute(json!({"name": "review"})).await });
    policy.entered.notified().await;

    fixture.write_skill("review", "Review changes.", "Use revision B.");
    let report = runtime.refresh_now().await;
    assert_eq!(report.updated(), &["review"]);
    policy.proceed.notify_one();
    worker
        .await
        .expect("activation worker")
        .expect("the old revision approval can complete");

    let prompt = runtime.compose_system_prompt("base").await;
    assert!(!prompt.contains("## Active skill: `review`"));
    assert!(!prompt.contains("Use revision B."));
}

#[tokio::test]
async fn activated_skill_resource_is_read_through_scoped_tool() {
    let fixture = SkillFixture::new();
    fixture.write_skill("research", "Research a topic.", "Read the reference.");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let runtime = fixture.runtime(registry);
    runtime.compose_system_prompt("base").await;
    let tools = runtime.tools();
    let activate = tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist");
    let read = tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::READ_RESOURCE_TOOL_NAME)
        .expect("resource tool should exist");

    let inactive = read
        .execute(json!({"skill": "research", "path": "references/details.md"}))
        .await;
    assert!(inactive.is_err());
    activate
        .execute(json!({"name": "research"}))
        .await
        .expect("skill should activate");
    let resource = read
        .execute(json!({"skill": "research", "path": "references/details.md"}))
        .await
        .expect("resource should be readable");
    assert_eq!(resource["content"], "Reference for research");

    fs::write(
        fixture.root.join("research/references/details.md"),
        "changed after discovery",
    )
    .expect("resource mutation");
    let changed = read
        .execute(json!({"skill": "research", "path": "references/details.md"}))
        .await;
    assert!(
        changed.is_err(),
        "undiscovered resource bytes must never reach the model"
    );

    let traversal = read
        .execute(json!({"skill": "research", "path": "../outside.md"}))
        .await;
    assert!(traversal.is_err());
}

#[tokio::test]
async fn skill_policy_denial_is_fail_closed_and_observable() {
    let fixture = SkillFixture::new();
    fixture.write_skill("research", "Research a topic.", "Read the reference.");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let configuration = SkillConfiguration::new(
        registry,
        Arc::new(SkillSession::new()),
        Arc::new(DenyActivationPolicy),
    );
    let (events, mut receiver) = tokio::sync::mpsc::channel(16);
    let runtime = SkillRuntime::new(
        configuration,
        SkillRuntimeIdentity::new(Uuid::new_v4(), Some(Uuid::new_v4()), Some(events)),
        Arc::new(SilentSkillLifecycle),
    );
    runtime.compose_system_prompt("base").await;
    let activate = runtime
        .tools()
        .into_iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist");

    let denied = activate.execute(json!({"name": "research"})).await;
    assert!(denied.is_err());
    assert!(!runtime.configuration().session().is_active("research"));

    let mut requested = false;
    let mut failed = false;
    while let Ok(event) = receiver.try_recv() {
        match event {
            Event::Skill {
                event:
                    autoagents_protocol::SkillEvent {
                        event: SkillEventKind::ActivationRequested { .. },
                        ..
                    },
            } => requested = true,
            Event::Skill {
                event:
                    autoagents_protocol::SkillEvent {
                        event: SkillEventKind::OperationFailed { .. },
                        ..
                    },
            } => failed = true,
            _ => {}
        }
    }
    assert!(requested);
    assert!(failed);
}

#[tokio::test]
async fn skill_lifecycle_receives_catalog_activation_resource_and_deactivation_events() {
    let fixture = SkillFixture::new();
    fixture.write_skill("research", "Research a topic.", "Read the reference.");
    let configuration = fixture.configuration(Arc::new(SkillRegistry::from_directory(
        fixture.root(),
        TEST_REFRESH,
    )));
    let lifecycle = Arc::new(RecordingSkillLifecycle::default());
    let runtime = SkillRuntime::new(
        configuration,
        SkillRuntimeIdentity::new(Uuid::new_v4(), Some(Uuid::new_v4()), None),
        lifecycle.clone(),
    );
    runtime.compose_system_prompt("base").await;
    let tools = runtime.tools();
    tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist")
        .execute(json!({"name": "research"}))
        .await
        .expect("activation should succeed");
    tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::READ_RESOURCE_TOOL_NAME)
        .expect("resource tool should exist")
        .execute(json!({"skill": "research", "path": "references/details.md"}))
        .await
        .expect("resource read should succeed");
    tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::DEACTIVATE_TOOL_NAME)
        .expect("deactivation tool should exist")
        .execute(json!({"name": "research"}))
        .await
        .expect("deactivation should succeed");

    let observed = lifecycle.observed();
    assert!(matches!(observed[0], SkillEventKind::CatalogChanged { .. }));
    assert!(
        observed
            .iter()
            .any(|event| matches!(event, SkillEventKind::ActivationRequested { .. }))
    );
    assert!(
        observed
            .iter()
            .any(|event| matches!(event, SkillEventKind::Activated { .. }))
    );
    assert!(
        observed
            .iter()
            .any(|event| matches!(event, SkillEventKind::ResourceAccessRequested { .. }))
    );
    assert!(
        observed
            .iter()
            .any(|event| matches!(event, SkillEventKind::ResourceAccessed { .. }))
    );
    assert!(
        observed
            .iter()
            .any(|event| matches!(event, SkillEventKind::Deactivated { .. }))
    );
}

#[tokio::test]
async fn explicit_deactivation_removes_instructions_from_following_prompts() {
    let fixture = SkillFixture::new();
    fixture.write_skill("research", "Research a topic.", "Read the reference.");
    let runtime = fixture.runtime(Arc::new(SkillRegistry::from_directory(
        fixture.root(),
        TEST_REFRESH,
    )));
    runtime.compose_system_prompt("base").await;
    let tools = runtime.tools();
    tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist")
        .execute(json!({"name": "research"}))
        .await
        .expect("activation should succeed");
    tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::DEACTIVATE_TOOL_NAME)
        .expect("deactivation tool should exist")
        .execute(json!({"name": "research"}))
        .await
        .expect("deactivation should succeed");

    let prompt = runtime.compose_system_prompt("base").await;
    assert!(!prompt.contains("## Active skill: `research`"));
}

#[tokio::test]
async fn invalid_live_edit_retains_last_valid_skill_until_repaired() {
    let fixture = SkillFixture::new();
    fixture.write_skill(
        "research",
        "Research a topic.",
        "Use the valid instructions.",
    );
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    registry.refresh().await;
    let original = registry.get("research").expect("valid skill should load");

    fixture.overwrite_document("research", "not valid frontmatter");
    let invalid = registry.refresh_now().await;
    assert!(!invalid.diagnostics().is_empty());
    let retained = registry
        .get("research")
        .expect("last valid skill should remain");
    assert!(Arc::ptr_eq(&original, &retained));

    fixture.overwrite_document(
        "research",
        "---\nname: research\ndescription: Research a topic.\n---\n\nUse repaired instructions.\n",
    );
    let repaired = registry.refresh_now().await;
    assert_eq!(repaired.updated(), &["research"]);
    assert_eq!(
        registry
            .get("research")
            .expect("repaired skill should load")
            .instructions(),
        "Use repaired instructions."
    );
}

#[cfg(feature = "skill-watch")]
#[tokio::test]
async fn forced_refresh_bypasses_watcher_debounce_for_resource_changes() {
    let fixture = SkillFixture::new();
    fixture.write_skill("resources", "Resource test.", "Read the resources.");
    let registry = SkillRegistry::from_directory(
        fixture.root(),
        SkillRefreshStrategy::Watch {
            debounce: std::time::Duration::from_secs(60),
            fallback_interval: std::time::Duration::from_secs(60),
        },
    );
    registry.refresh().await;

    fs::write(
        fixture.root.join("resources/references/just-added.md"),
        "new resource",
    )
    .expect("new resource");
    let report = registry.refresh_now().await;

    assert_eq!(report.updated(), &["resources"]);
    assert!(
        registry
            .get("resources")
            .expect("resource skill")
            .resources()
            .contains(&PathBuf::from("references/just-added.md"))
    );
}

#[tokio::test]
async fn manual_refresh_ignores_changes_until_explicitly_invalidated() {
    let fixture = SkillFixture::new();
    fixture.write_skill("initial", "Initial skill.", "Initial instructions.");
    let registry = SkillRegistry::from_directory(fixture.root(), SkillRefreshStrategy::Manual);
    registry.refresh().await;

    fixture.write_skill("added", "Added skill.", "Added instructions.");
    let unchanged = registry.refresh().await;
    assert!(!unchanged.changed());
    assert!(registry.get("added").is_none());

    let refreshed = registry.refresh_now().await;
    assert_eq!(refreshed.added(), &["added"]);
    assert!(registry.get("added").is_some());
}

#[tokio::test]
async fn zero_interval_poll_refreshes_at_the_next_registry_boundary() {
    let fixture = SkillFixture::new();
    let registry = SkillRegistry::from_directory(fixture.root(), TEST_REFRESH);
    registry.refresh().await;

    fixture.write_skill("added", "Added skill.", "Added instructions.");
    let refreshed = registry.refresh().await;

    assert_eq!(refreshed.added(), &["added"]);
}

#[cfg(not(feature = "skill-watch"))]
#[tokio::test]
async fn watch_strategy_requires_the_optional_cargo_feature() {
    let fixture = SkillFixture::new();
    let source = DirectorySkillSource::new(
        fixture.root(),
        SkillRefreshStrategy::Watch {
            debounce: std::time::Duration::from_millis(150),
            fallback_interval: std::time::Duration::from_secs(30),
        },
    );

    let error = source
        .discover()
        .await
        .expect_err("watching must fail explicitly when it was not compiled");

    assert!(matches!(error, SkillError::WatchUnavailable));
}

#[tokio::test]
async fn resource_content_updates_change_revision_and_deactivate_the_skill() {
    let fixture = SkillFixture::new();
    fixture.write_skill("resources", "Resource test.", "Read the resources.");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let runtime = fixture.runtime(Arc::clone(&registry));
    runtime.compose_system_prompt("base").await;
    let activation = runtime
        .tools()
        .into_iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool");
    activation
        .execute(json!({"name": "resources"}))
        .await
        .expect("skill activation");
    let original_revision = registry.get("resources").expect("skill").revision();

    fs::write(
        fixture.root.join("resources/references/details.md"),
        "replacement resource content",
    )
    .expect("resource update");
    let report = runtime.refresh_now().await;

    assert_eq!(report.updated(), &["resources"]);
    assert_ne!(
        original_revision,
        registry.get("resources").expect("updated skill").revision()
    );
    assert!(
        !runtime
            .compose_system_prompt("base")
            .await
            .contains("## Active skill: `resources`")
    );
}

#[test]
fn allowed_tools_are_parsed_into_typed_selectors() {
    let fixture = SkillFixture::new();
    let directory = fixture.root.join("tool-aware");
    fs::create_dir_all(&directory).expect("skill directory should be created");
    let document = "---\nname: tool-aware\ndescription: Uses selected tools.\nallowed-tools: Bash(git:*) Read mcp.server/tool\n---\n\nUse only declared capabilities.\n";
    let skill =
        super::Skill::from_document(directory.join("SKILL.md"), directory, document, Vec::new())
            .expect("tool selectors should parse");

    let selectors = skill.metadata().allowed_tools();
    assert_eq!(selectors.len(), 3);
    assert_eq!(selectors[0].tool_name(), "Bash");
    assert_eq!(selectors[0].qualifier(), Some("git:*"));
    assert_eq!(selectors[2].tool_name(), "mcp.server/tool");
}

#[tokio::test]
async fn oversized_skill_resources_are_rejected_without_full_reads() {
    let fixture = SkillFixture::new();
    fixture.write_skill("research", "Research a topic.", "Read the reference.");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let runtime = fixture.runtime_with_resource_limit(registry, 4);
    runtime.compose_system_prompt("base").await;
    let tools = runtime.tools();
    let activate = tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist");
    let read = tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::READ_RESOURCE_TOOL_NAME)
        .expect("resource tool should exist");
    activate
        .execute(json!({"name": "research"}))
        .await
        .expect("skill should activate");

    let error = read
        .execute(json!({"skill": "research", "path": "references/details.md"}))
        .await
        .expect_err("oversized resource should be rejected");

    assert!(error.to_string().contains("exceeds the 4 byte limit"));
}

#[test]
fn reserved_skill_tool_names_are_rejected() {
    let tools: Vec<Box<dyn ToolT>> = vec![Box::new(ReservedNameTool)];

    let error = SkillRuntime::validate_agent_tools(&tools)
        .expect_err("reserved skill tool name should be rejected");

    assert!(matches!(error, SkillError::ToolNameConflict(_)));
}

#[tokio::test]
async fn discovery_limits_skill_document_and_resource_counts() {
    let fixture = SkillFixture::new();
    fixture.write_skill("limited", "A limited skill.", "instructions");
    let limits = SkillDiscoveryLimits {
        max_skills: 1,
        max_skill_file_bytes: 10,
        max_resources_per_skill: 0,
        max_resource_depth: 1,
        ..SkillDiscoveryLimits::default()
    };
    let registry = SkillRegistry::new(vec![Arc::new(
        DirectorySkillSource::new(fixture.root(), TEST_REFRESH).with_limits(limits),
    )]);

    let oversized = registry.refresh().await;
    assert_eq!(oversized.discovered(), 0);
    assert!(oversized.diagnostics()[0].message().contains("byte limit"));

    let aggregate_limited = SkillRegistry::new(vec![Arc::new(
        DirectorySkillSource::new(fixture.root(), TEST_REFRESH).with_limits(SkillDiscoveryLimits {
            max_total_skill_file_bytes: 1,
            ..SkillDiscoveryLimits::default()
        }),
    )]);
    let aggregate = aggregate_limited.refresh().await;
    assert_eq!(aggregate.discovered(), 0);
    assert!(
        aggregate
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message().contains("aggregate"))
    );

    let resource_limited = SkillRegistry::new(vec![Arc::new(
        DirectorySkillSource::new(fixture.root(), TEST_REFRESH).with_limits(SkillDiscoveryLimits {
            max_resources_per_skill: 0,
            ..SkillDiscoveryLimits::default()
        }),
    )]);
    resource_limited.refresh().await;
    assert!(
        resource_limited
            .get("limited")
            .expect("skill should be discovered")
            .resources()
            .is_empty()
    );

    let resource_size_limited = SkillRegistry::new(vec![Arc::new(
        DirectorySkillSource::new(fixture.root(), TEST_REFRESH).with_limits(SkillDiscoveryLimits {
            max_resource_file_bytes: 4,
            ..SkillDiscoveryLimits::default()
        }),
    )]);
    let resource_size_report = resource_size_limited.refresh().await;
    assert_eq!(resource_size_report.discovered(), 0);
    assert!(
        resource_size_report
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message().contains("resource exceeds"))
    );
}

#[tokio::test]
async fn higher_precedence_source_overrides_lower_precedence_skill() {
    let lower = SkillFixture::new();
    lower.write_skill("review", "Lower precedence.", "lower");
    let higher = SkillFixture::new();
    higher.write_skill("review", "Higher precedence.", "higher");
    let sources: Vec<Arc<dyn SkillSource>> = vec![
        Arc::new(DirectorySkillSource::new(lower.root(), TEST_REFRESH).with_precedence(10)),
        Arc::new(DirectorySkillSource::new(higher.root(), TEST_REFRESH).with_precedence(20)),
    ];
    let registry = SkillRegistry::new(sources);

    let report = registry.refresh().await;

    assert_eq!(report.discovered(), 1);
    assert_eq!(
        registry
            .get("review")
            .expect("review skill should exist")
            .metadata()
            .description(),
        "Higher precedence."
    );
}

#[tokio::test]
async fn malformed_skill_is_diagnosed_without_hiding_valid_skills() {
    let fixture = SkillFixture::new();
    fixture.write_skill("valid-skill", "A valid skill.", "valid");
    let malformed = fixture.root().join("malformed");
    fs::create_dir_all(&malformed).expect("malformed directory should be created");
    fs::write(
        malformed.join("SKILL.md"),
        "---\nname: Wrong_Name\ndescription: invalid\n---\nbody",
    )
    .expect("malformed skill should be written");
    let registry = SkillRegistry::from_directory(fixture.root(), TEST_REFRESH);

    let report = registry.refresh().await;

    assert_eq!(report.discovered(), 1);
    assert_eq!(report.diagnostics().len(), 1);
    assert!(registry.get("valid-skill").is_some());

    let unchanged = registry.refresh().await;
    assert!(unchanged.diagnostics().is_empty());
}

#[tokio::test]
async fn unchanged_refresh_reuses_skill_and_snapshot_allocations() {
    let fixture = SkillFixture::new();
    fixture.write_skill("cached-skill", "A cached skill.", "cached instructions");
    let registry = SkillRegistry::from_directory(fixture.root(), TEST_REFRESH);
    registry.refresh().await;
    let initial_snapshot = registry.snapshot();
    let initial_skill = registry
        .get("cached-skill")
        .expect("cached skill should exist");

    let report = registry.refresh().await;
    let refreshed_snapshot = registry.snapshot();
    let refreshed_skill = registry
        .get("cached-skill")
        .expect("cached skill should still exist");

    assert!(!report.changed());
    assert!(Arc::ptr_eq(&initial_snapshot, &refreshed_snapshot));
    assert!(Arc::ptr_eq(&initial_skill, &refreshed_skill));

    let forced = registry.refresh_now().await;
    let verified_skill = registry
        .get("cached-skill")
        .expect("verified skill should still exist");

    assert!(!forced.changed());
    assert!(Arc::ptr_eq(&initial_skill, &verified_skill));
}

#[tokio::test]
async fn concurrent_refresh_requests_share_one_discovery_pass() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(SkillRegistry::new(vec![Arc::new(CountingSkillSource {
        id: SkillSourceId::new("counting"),
        calls: Arc::clone(&calls),
    })]));
    let first = Arc::clone(&registry);
    let second = Arc::clone(&registry);

    let (first_report, second_report) = tokio::join!(first.refresh(), second.refresh());

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(first_report.generation(), second_report.generation());
}

#[tokio::test]
async fn forced_refresh_waits_for_and_follows_an_in_flight_discovery() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(SkillRegistry::new(vec![Arc::new(CountingSkillSource {
        id: SkillSourceId::new("counting"),
        calls: Arc::clone(&calls),
    })]));
    let first_registry = Arc::clone(&registry);
    let first = tokio::spawn(async move { first_registry.refresh().await });
    while calls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }

    registry.refresh_now().await;
    first.await.expect("initial refresh should complete");

    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[cfg(unix)]
#[tokio::test]
async fn symbolic_link_skill_document_is_rejected() {
    use std::os::unix::fs::symlink;

    let fixture = SkillFixture::new();
    let outside = fixture._temporary.path().join("outside-skill-document.md");
    fs::write(
        &outside,
        "---\nname: linked-skill\ndescription: Escapes the root.\n---\nbody",
    )
    .expect("outside document should be written");
    let skill_directory = fixture.root().join("linked-skill");
    fs::create_dir_all(&skill_directory).expect("skill directory should be created");
    symlink(&outside, skill_directory.join("SKILL.md")).expect("symlink should be created");
    let registry = SkillRegistry::from_directory(fixture.root(), TEST_REFRESH);

    let report = registry.refresh().await;

    assert_eq!(report.discovered(), 0);
    assert_eq!(report.diagnostics().len(), 1);
    assert!(report.diagnostics()[0].message().contains("symbolic link"));
}

#[cfg(unix)]
#[tokio::test]
async fn symbolic_link_skill_resource_is_not_readable() {
    use std::os::unix::fs::symlink;

    let fixture = SkillFixture::new();
    fixture.write_skill("research", "Research a topic.", "Read the reference.");
    let outside = fixture._temporary.path().join("outside-resource.md");
    fs::write(&outside, "outside secret").expect("outside resource should be written");
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let runtime = fixture.runtime(registry);
    runtime.compose_system_prompt("base").await;
    let tools = runtime.tools();
    let activate = tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::ACTIVATE_TOOL_NAME)
        .expect("activation tool should exist");
    let read = tools
        .iter()
        .find(|tool| tool.name() == SkillRuntime::READ_RESOURCE_TOOL_NAME)
        .expect("resource tool should exist");
    activate
        .execute(json!({"name": "research"}))
        .await
        .expect("skill should activate");

    let resource = fixture.root().join("research/references/details.md");
    fs::remove_file(&resource).expect("original resource should be removed");
    symlink(&outside, &resource).expect("resource symlink should be created");
    let result = read
        .execute(json!({"skill": "research", "path": "references/details.md"}))
        .await;

    assert!(result.is_err());
}

#[cfg(unix)]
#[tokio::test]
async fn contained_directory_source_rejects_a_symlinked_root_escape() {
    use std::os::unix::fs::symlink;

    let workspace = tempfile::tempdir().expect("workspace");
    let outside = SkillFixture::new();
    outside.write_skill("external", "External instructions.", "Do not load this.");
    fs::create_dir_all(workspace.path().join(".agents")).expect("agent directory");
    let source_root = workspace.path().join(".agents/skills");
    symlink(outside.root(), &source_root).expect("skill root symlink");

    let source = DirectorySkillSource::new(&source_root, TEST_REFRESH)
        .with_containment_root(workspace.path().to_path_buf());
    let error = source
        .discover()
        .await
        .expect_err("containment must reject a source root that resolves outside the workspace");

    assert!(error.to_string().contains("escapes containment root"));

    #[cfg(feature = "skill-watch")]
    {
        let watched_source = DirectorySkillSource::new(
            &source_root,
            SkillRefreshStrategy::Watch {
                debounce: std::time::Duration::from_millis(150),
                fallback_interval: std::time::Duration::from_secs(30),
            },
        )
        .with_containment_root(workspace.path().to_path_buf());
        let watched_error = watched_source
            .discover()
            .await
            .expect_err("watch setup must reject an escaped source before observing it");

        assert!(
            watched_error
                .to_string()
                .contains("escapes containment root")
        );
    }
}

#[test]
fn skill_revision_changes_with_document_or_resource_manifest() {
    let directory = PathBuf::from("revisioned");
    let first = super::Skill::from_document(
        directory.join("SKILL.md"),
        directory.clone(),
        "---\nname: revisioned\ndescription: Revision test.\n---\n\nFirst instructions.",
        vec![PathBuf::from("reference.md")],
    )
    .expect("first skill");
    let same = super::Skill::from_document(
        directory.join("SKILL.md"),
        directory.clone(),
        "---\nname: revisioned\ndescription: Revision test.\n---\n\nFirst instructions.",
        vec![PathBuf::from("reference.md")],
    )
    .expect("same skill");
    let changed_document = super::Skill::from_document(
        directory.join("SKILL.md"),
        directory.clone(),
        "---\nname: revisioned\ndescription: Revision test.\n---\n\nChanged instructions.",
        vec![PathBuf::from("reference.md")],
    )
    .expect("changed document");
    let changed_resources = super::Skill::from_document(
        directory.join("SKILL.md"),
        directory,
        "---\nname: revisioned\ndescription: Revision test.\n---\n\nFirst instructions.",
        vec![PathBuf::from("other.md")],
    )
    .expect("changed resources");

    assert_eq!(first.revision(), same.revision());
    assert_ne!(first.revision(), changed_document.revision());
    assert_ne!(first.revision(), changed_resources.revision());
}

#[test]
fn skill_model_exposes_metadata_paths_and_revision() {
    let directory = PathBuf::from("documented-skill");
    let skill = super::Skill::from_document(
        directory.join("SKILL.md"),
        directory.clone(),
        "---\nname: documented-skill\ndescription: A documented skill.\nlicense: Apache-2.0\ncompatibility: Native and wasm.\nmetadata:\n  author: autoagents\n---\n\nFollow the documented process.\n",
        vec![
            PathBuf::from("references/zeta.md"),
            PathBuf::from("references/alpha.md"),
            PathBuf::from("references/alpha.md"),
        ],
    )
    .expect("valid skill document");

    assert_eq!(skill.metadata().license(), Some("Apache-2.0"));
    assert_eq!(skill.metadata().compatibility(), Some("Native and wasm."));
    assert_eq!(
        skill
            .metadata()
            .metadata()
            .get("author")
            .map(String::as_str),
        Some("autoagents")
    );
    assert_eq!(skill.skill_file(), directory.join("SKILL.md"));
    assert_eq!(
        skill.resources(),
        &[
            PathBuf::from("references/alpha.md"),
            PathBuf::from("references/zeta.md"),
        ]
    );
    let revision = skill.revision().to_string();
    assert_eq!(revision.len(), 64);
    assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
}

#[test]
fn skill_document_validation_rejects_invalid_metadata_and_body() {
    fn parse(name: &str, document: &str) -> Result<super::Skill, SkillError> {
        let directory = PathBuf::from(name);
        super::Skill::from_document(directory.join("SKILL.md"), directory, document, Vec::new())
    }

    let cases = [
        (
            "empty-name",
            "---\nname: ''\ndescription: Description.\n---\nbody",
            "name must contain",
        ),
        (
            "leading-hyphen",
            "---\nname: -leading-hyphen\ndescription: Description.\n---\nbody",
            "name must use lowercase",
        ),
        (
            "directory-name",
            "---\nname: another-name\ndescription: Description.\n---\nbody",
            "must match parent directory",
        ),
        (
            "empty-description",
            "---\nname: empty-description\ndescription: '   '\n---\nbody",
            "description must contain",
        ),
        (
            "empty-compatibility",
            "---\nname: empty-compatibility\ndescription: Description.\ncompatibility: '   '\n---\nbody",
            "compatibility must contain",
        ),
        (
            "invalid-yaml",
            "---\nname: [invalid\ndescription: Description.\n---\nbody",
            "invalid YAML frontmatter",
        ),
        (
            "empty-instructions",
            "---\nname: empty-instructions\ndescription: Description.\n---\n   ",
            "instructions must not be empty",
        ),
        (
            "missing-closing-delimiter",
            "---\nname: missing-closing-delimiter\ndescription: Description.\nbody",
            "missing its closing delimiter",
        ),
    ];

    for (directory, document, expected) in cases {
        let error = parse(directory, document).expect_err("document should be rejected");
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error}"
        );
    }

    let valid_document = "---\nname: invalid-resource\ndescription: Description.\n---\nbody";
    let invalid_resource = super::Skill::from_document(
        "invalid-resource/SKILL.md",
        "invalid-resource",
        valid_document,
        vec![PathBuf::from("../outside.md")],
    )
    .expect_err("parent resource path should be rejected");
    assert!(invalid_resource.to_string().contains("resource path"));
}

#[cfg(unix)]
#[test]
fn skill_document_rejects_a_non_utf8_parent_directory() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let directory = PathBuf::from(OsString::from_vec(vec![0xff]));
    let error = super::Skill::from_document(
        directory.join("SKILL.md"),
        directory,
        "---\nname: valid-name\ndescription: Description.\n---\nbody",
        Vec::new(),
    )
    .expect_err("non-UTF-8 parent should be rejected");

    assert!(
        error
            .to_string()
            .contains("parent directory is not valid UTF-8")
    );
}

#[tokio::test]
async fn skill_policy_requests_and_configuration_expose_their_context() {
    let directory = PathBuf::from("request-skill");
    let skill = Arc::new(
        super::Skill::from_document(
            directory.join("SKILL.md"),
            directory,
            "---\nname: request-skill\ndescription: Request context.\n---\nbody",
            vec![PathBuf::from("reference.md")],
        )
        .expect("request skill"),
    );
    let actor_id = Uuid::new_v4();
    let submission_id = Uuid::new_v4();
    let session_id = Uuid::new_v4();
    let activation =
        SkillActivationRequest::new(actor_id, submission_id, session_id, Arc::clone(&skill));
    assert_eq!(activation.actor_id(), actor_id);
    assert_eq!(activation.submission_id(), submission_id);
    assert_eq!(activation.session_id(), session_id);
    assert_eq!(activation.skill().metadata().name(), "request-skill");

    let resource = SkillResourceRequest::new(
        actor_id,
        submission_id,
        session_id,
        Arc::clone(&skill),
        PathBuf::from("reference.md"),
    );
    assert_eq!(resource.actor_id(), actor_id);
    assert_eq!(resource.submission_id(), submission_id);
    assert_eq!(resource.session_id(), session_id);
    assert_eq!(resource.skill().revision(), skill.revision());
    assert_eq!(resource.path(), Path::new("reference.md"));

    let registry = Arc::new(SkillRegistry::new(Vec::new()));
    let session = Arc::new(SkillSession::new());
    let policy: Arc<dyn SkillPolicy> = Arc::new(TrustedSkillPolicy);
    let configuration = SkillConfiguration::new(
        Arc::clone(&registry),
        Arc::clone(&session),
        Arc::clone(&policy),
    );
    assert!(Arc::ptr_eq(&configuration.registry(), &registry));
    assert!(Arc::ptr_eq(&configuration.session(), &session));
    assert!(Arc::ptr_eq(&configuration.policy(), &policy));
    let debug = format!("{configuration:?}");
    assert!(debug.contains("SkillConfiguration"));
    assert!(debug.contains("max_resource_bytes"));

    policy
        .authorize_activation(&activation)
        .await
        .expect("trusted activation");
    policy
        .authorize_resource_read(&resource)
        .await
        .expect("trusted resource read");
}

#[tokio::test]
async fn directory_source_handles_missing_roots_containment_and_entry_limits() {
    let fixture = SkillFixture::new();
    fixture.write_skill("first", "First skill.", "First instructions.");
    fixture.write_skill("second", "Second skill.", "Second instructions.");
    fs::create_dir_all(fixture.root().join("without-document")).expect("empty skill directory");

    let limited =
        DirectorySkillSource::new(fixture.root(), TEST_REFRESH).with_limits(SkillDiscoveryLimits {
            max_skills: 2,
            ..SkillDiscoveryLimits::default()
        });
    let limited_snapshot = limited.discover().await.expect("limited discovery");
    assert_eq!(limited_snapshot.skills.len(), 2);
    assert!(
        limited_snapshot
            .diagnostics
            .iter()
            .any(|message| message.contains("only the first 2 are scanned"))
    );

    let contained = DirectorySkillSource::new(fixture.root(), TEST_REFRESH)
        .with_containment_root(fixture._temporary.path());
    assert_eq!(
        contained
            .discover()
            .await
            .expect("contained discovery")
            .skills
            .len(),
        2
    );

    let missing_containment = fixture._temporary.path().join("missing-containment");
    let containment_error = DirectorySkillSource::new(fixture.root(), TEST_REFRESH)
        .with_containment_root(missing_containment)
        .discover()
        .await
        .expect_err("missing containment root should fail");
    assert!(
        containment_error
            .to_string()
            .contains("cannot resolve containment root")
    );

    let disappearing_root = fixture._temporary.path().join("disappearing-skills");
    fs::create_dir_all(&disappearing_root).expect("disappearing root");
    let source = DirectorySkillSource::new(&disappearing_root, SkillRefreshStrategy::Manual);
    assert!(
        source
            .discover()
            .await
            .expect("initial discovery")
            .skills
            .is_empty()
    );
    fs::remove_dir_all(&disappearing_root).expect("remove disappearing root");
    source.invalidate();
    assert!(
        source
            .discover()
            .await
            .expect("missing root is empty")
            .skills
            .is_empty()
    );
}

#[tokio::test]
async fn directory_source_enforces_the_aggregate_resource_budget() {
    let fixture = SkillFixture::new();
    fixture.write_skill("resources", "Resource limits.", "Read both resources.");
    fs::write(
        fixture.root().join("resources/references/second.md"),
        "second resource",
    )
    .expect("second resource");
    let source =
        DirectorySkillSource::new(fixture.root(), TEST_REFRESH).with_limits(SkillDiscoveryLimits {
            max_total_resource_bytes: 1,
            ..SkillDiscoveryLimits::default()
        });

    let snapshot = source.discover().await.expect("bounded discovery");

    assert!(snapshot.skills.is_empty());
    assert!(
        snapshot
            .diagnostics
            .iter()
            .any(|message| message.contains("aggregate limit"))
    );
}

#[test]
fn skill_tool_selector_exposes_expression_and_rejects_invalid_forms() {
    let selector = super::SkillToolSelector::parse("Bash(git:*)").expect("valid selector");
    assert_eq!(selector.expression(), "Bash(git:*)");

    for invalid in ["", "two tools", "Bash(", "bad$name"] {
        let error = super::SkillToolSelector::parse(invalid)
            .expect_err("invalid selector should be rejected");
        assert!(matches!(error, SkillError::InvalidToolSelector(_)));
    }
}

#[test]
fn skill_runtime_exposes_debug_configuration_and_serialized_tools() {
    let fixture = SkillFixture::new();
    let registry = Arc::new(SkillRegistry::from_directory(fixture.root(), TEST_REFRESH));
    let runtime = fixture.runtime(registry);
    let (events, _receiver) = tokio::sync::mpsc::channel(1);
    let identity = SkillRuntimeIdentity::new(Uuid::new_v4(), Some(Uuid::new_v4()), Some(events));

    let debug = format!("{identity:?}");
    assert!(debug.contains("SkillRuntimeIdentity"));
    assert!(debug.contains("configured"));
    assert_eq!(runtime.serialized_tools().len(), 3);
    assert_eq!(runtime.configuration().max_resource_bytes(), 1024 * 1024);
}
