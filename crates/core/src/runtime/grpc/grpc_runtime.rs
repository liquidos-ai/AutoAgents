use crate::{
    agent::RunnableAgent,
    error::Error,
    protocol::{AgentID, Event, RuntimeID},
    runtime::{Runtime, RuntimeError, Task},
};
use async_trait::async_trait;
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use uuid::Uuid;

use super::proto::{
    agent_runtime_server::{AgentRuntime as AgentRuntimeService, AgentRuntimeServer},
    Event as ProtoEvent, EventStreamRequest, HeartbeatRequest, HeartbeatResponse,
    PublishMessageRequest, PublishMessageResponse, RegisterAgentRequest, RegisterAgentResponse,
    SendMessageRequest, SendMessageResponse, SubscribeRequest, SubscribeResponse,
};

/// Configuration for the gRPC runtime
#[derive(Debug, Clone)]
pub struct GrpcRuntimeConfig {
    /// Address to bind the gRPC server to
    pub bind_addr: SocketAddr,
    /// Maximum message size in bytes (default: 4MB)
    pub max_message_size: usize,
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Channel buffer size for internal communication
    pub channel_buffer_size: usize,
}

impl Default for GrpcRuntimeConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:50051".parse().unwrap(),
            max_message_size: 4 * 1024 * 1024, // 4MB
            max_connections: 100,
            channel_buffer_size: 1000,
        }
    }
}

/// Agent connection information
struct AgentConnection {
    agent_id: AgentID,
    agent_name: String,
    event_tx: mpsc::Sender<ProtoEvent>,
    subscriptions: Vec<String>,
}

/// gRPC-based runtime for distributed agent systems
#[derive(Clone)]
pub struct GrpcRuntime {
    /// Runtime ID
    pub id: RuntimeID,
    /// Configuration
    config: GrpcRuntimeConfig,
    /// External event sender (for broadcasting to all listeners)
    external_tx: mpsc::Sender<Event>,
    /// External event receiver
    external_rx: Arc<RwLock<Option<mpsc::Receiver<Event>>>>,
    /// Internal event sender (for routing between agents)
    internal_tx: mpsc::Sender<Event>,
    /// Internal event receiver
    internal_rx: Arc<RwLock<Option<mpsc::Receiver<Event>>>>,
    /// Connected agents
    connections: Arc<RwLock<HashMap<AgentID, AgentConnection>>>,
    /// Local agents (in-process)
    local_agents: Arc<RwLock<HashMap<AgentID, Arc<dyn RunnableAgent>>>>,
    /// Topic subscriptions
    subscriptions: Arc<RwLock<HashMap<String, Vec<AgentID>>>>,
    /// Shutdown signal
    shutdown_tx: Arc<RwLock<Option<oneshot::Sender<()>>>>,
}

impl GrpcRuntime {
    /// Create a new gRPC runtime with the given configuration
    pub async fn new(config: GrpcRuntimeConfig) -> Result<Arc<Self>, Error> {
        let (external_tx, external_rx) = mpsc::channel(config.channel_buffer_size);
        let (internal_tx, internal_rx) = mpsc::channel(config.channel_buffer_size);

        Ok(Arc::new(Self {
            id: Uuid::new_v4(),
            config,
            external_tx,
            external_rx: Arc::new(RwLock::new(Some(external_rx))),
            internal_tx,
            internal_rx: Arc::new(RwLock::new(Some(internal_rx))),
            connections: Arc::new(RwLock::new(HashMap::new())),
            local_agents: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx: Arc::new(RwLock::new(None)),
        }))
    }

    /// Start the gRPC server
    async fn start_server(&self) -> Result<(), Error> {
        let service = GrpcRuntimeServiceImpl {
            runtime_id: self.id,
            connections: self.connections.clone(),
            subscriptions: self.subscriptions.clone(),
            internal_tx: self.internal_tx.clone(),
            external_tx: self.external_tx.clone(),
        };

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        *self.shutdown_tx.write().await = Some(shutdown_tx);

        let addr = self.config.bind_addr;
        let max_message_size = self.config.max_message_size;

        tokio::spawn(async move {
            let svc = AgentRuntimeServer::new(service)
                .max_decoding_message_size(max_message_size)
                .max_encoding_message_size(max_message_size);

            let _ = Server::builder()
                .add_service(svc)
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        Ok(())
    }

    /// Process internal events and route them appropriately
    async fn process_internal_messages(&self) {
        let mut rx_lock = self.internal_rx.write().await;
        if let Some(mut internal_rx) = rx_lock.take() {
            drop(rx_lock);
            while let Some(event) = internal_rx.recv().await {
                // Broadcast to external listeners
                let _ = self.external_tx.send(event.clone()).await;

                // Route based on event type
                match &event {
                    Event::PublishMessage { topic, message } => {
                        self.route_to_subscribers(topic, message).await;
                    }
                    Event::SendMessage { agent_id, message } => {
                        self.route_to_agent(agent_id, message).await;
                    }
                    Event::NewTask { agent_id, task } => {
                        self.send_task_to_agent(agent_id, task).await;
                    }
                    _ => {
                        // Other events are just broadcast
                    }
                }
            }
        }
    }

    /// Route message to all subscribers of a topic
    async fn route_to_subscribers(&self, topic: &str, message: &str) {
        let subscriptions = self.subscriptions.read().await;
        if let Some(subscribers) = subscriptions.get(topic) {
            for agent_id in subscribers {
                // Create a task for this message
                let task = Task::new(message.to_string(), Some(*agent_id));

                // Send to local agents first
                let local_agents = self.local_agents.read().await;
                if let Some(agent) = local_agents.get(agent_id) {
                    let agent = agent.clone();
                    let internal_tx = self.internal_tx.clone();

                    // Send NewTask event to external channel
                    let _ = self
                        .external_tx
                        .send(Event::NewTask {
                            agent_id: *agent_id,
                            task: task.clone(),
                        })
                        .await;

                    // Spawn agent execution with the task
                    agent.clone().spawn_task(task.clone(), internal_tx);
                } else {
                    // Send to remote agents
                    let connections = self.connections.read().await;
                    if let Some(connection) = connections.get(agent_id) {
                        let _ = connection
                            .event_tx
                            .send(ProtoEvent {
                                event_type: Some(super::proto::event::EventType::NewTask(
                                    super::proto::NewTaskEvent {
                                        agent_id: Some(super::proto::AgentId {
                                            uuid: agent_id.to_string(),
                                        }),
                                        task: Some(super::proto::Task {
                                            prompt: task.prompt,
                                            submission_id: task.submission_id.to_string(),
                                            completed: task.completed,
                                            result: task.result.as_ref().map(|v| v.to_string()),
                                            agent_id: None,
                                        }),
                                    },
                                )),
                            })
                            .await;
                    }
                }
            }
        }
    }

    /// Route message directly to a specific agent
    async fn route_to_agent(&self, agent_id: &AgentID, message: &str) {
        // Check remote connections first
        let connections = self.connections.read().await;
        if let Some(connection) = connections.get(agent_id) {
            let _ = connection
                .event_tx
                .send(ProtoEvent {
                    event_type: Some(super::proto::event::EventType::SendMessage(
                        super::proto::SendMessageEvent {
                            agent_id: Some(super::proto::AgentId {
                                uuid: agent_id.to_string(),
                            }),
                            message: message.to_string(),
                        },
                    )),
                })
                .await;
            return;
        }
        drop(connections);

        // Check local agents
        let local_agents = self.local_agents.read().await;
        if let Some(agent) = local_agents.get(agent_id) {
            let agent = agent.clone();
            let internal_tx = self.internal_tx.clone();

            // Create a task from the message
            let task = Task::new(message.to_string(), Some(*agent_id));

            // Send NewTask event to external channel
            let _ = self
                .external_tx
                .send(Event::NewTask {
                    agent_id: *agent_id,
                    task: task.clone(),
                })
                .await;

            // Spawn agent execution with the task
            agent.clone().spawn_task(task, internal_tx);
        }
    }

    /// Send a task to a specific agent
    async fn send_task_to_agent(&self, agent_id: &AgentID, task: &Task) {
        // Check remote connections first
        let connections = self.connections.read().await;
        if let Some(connection) = connections.get(agent_id) {
            let _ = connection
                .event_tx
                .send(ProtoEvent {
                    event_type: Some(super::proto::event::EventType::NewTask(
                        super::proto::NewTaskEvent {
                            agent_id: Some(super::proto::AgentId {
                                uuid: agent_id.to_string(),
                            }),
                            task: Some(super::proto::Task {
                                prompt: task.prompt.clone(),
                                submission_id: task.submission_id.to_string(),
                                completed: task.completed,
                                result: task.result.as_ref().map(|v| v.to_string()),
                                agent_id: None,
                            }),
                        },
                    )),
                })
                .await;
            return;
        }
        drop(connections);

        // Check local agents
        let local_agents = self.local_agents.read().await;
        if let Some(agent) = local_agents.get(agent_id) {
            let agent = agent.clone();
            let internal_tx = self.internal_tx.clone();

            // Send NewTask event to external channel
            let _ = self
                .external_tx
                .send(Event::NewTask {
                    agent_id: *agent_id,
                    task: task.clone(),
                })
                .await;

            // Spawn agent execution with the task
            agent.clone().spawn_task(task.clone(), internal_tx);
        }
    }
}

#[async_trait]
impl Runtime for GrpcRuntime {
    fn id(&self) -> RuntimeID {
        self.id
    }

    async fn send_message(&self, message: String, agent_id: AgentID) -> Result<(), Error> {
        self.internal_tx
            .send(Event::SendMessage { agent_id, message })
            .await
            .map_err(|e| Error::RuntimeError(RuntimeError::EventError(e)))
    }

    async fn publish_message(&self, message: String, topic: String) -> Result<(), Error> {
        self.internal_tx
            .send(Event::PublishMessage { topic, message })
            .await
            .map_err(|e| Error::RuntimeError(RuntimeError::EventError(e)))
    }

    async fn subscribe(&self, agent_id: AgentID, topic: String) -> Result<(), Error> {
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions
            .entry(topic)
            .or_insert_with(Vec::new)
            .push(agent_id);
        Ok(())
    }

    async fn register_agent(&self, agent: Arc<dyn RunnableAgent>) -> Result<(), Error> {
        let agent_id = agent.id();

        // Just store the agent reference
        let mut local_agents = self.local_agents.write().await;
        local_agents.insert(agent_id, agent);

        Ok(())
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        let mut rx_lock = self.external_rx.write().await;
        rx_lock.take().map(ReceiverStream::new)
    }

    async fn run(&self) -> Result<(), Error> {
        // Start gRPC server
        self.start_server().await?;

        // Process internal messages
        self.process_internal_messages().await;

        Ok(())
    }

    async fn stop(&self) -> Result<(), Error> {
        // Stop gRPC server
        let mut shutdown_lock = self.shutdown_tx.write().await;
        if let Some(shutdown_tx) = shutdown_lock.take() {
            let _ = shutdown_tx.send(());
        }

        Ok(())
    }
}

impl std::fmt::Debug for GrpcRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcRuntime")
            .field("id", &self.id)
            .field("config", &self.config)
            .finish()
    }
}

/// gRPC service implementation
struct GrpcRuntimeServiceImpl {
    runtime_id: RuntimeID,
    connections: Arc<RwLock<HashMap<AgentID, AgentConnection>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<AgentID>>>>,
    internal_tx: mpsc::Sender<Event>,
    external_tx: mpsc::Sender<Event>,
}

#[tonic::async_trait]
impl AgentRuntimeService for GrpcRuntimeServiceImpl {
    async fn register_agent(
        &self,
        request: Request<RegisterAgentRequest>,
    ) -> Result<Response<RegisterAgentResponse>, Status> {
        let req = request.into_inner();
        let agent_id = req
            .agent_id
            .ok_or_else(|| Status::invalid_argument("agent_id is required"))?;
        let agent_uuid = Uuid::parse_str(&agent_id.uuid)
            .map_err(|_| Status::invalid_argument("Invalid agent UUID"))?;

        // Create event channel for this agent
        let (event_tx, _event_rx) = mpsc::channel(100);

        let connection = AgentConnection {
            agent_id: agent_uuid,
            agent_name: req.agent_name.clone(),
            event_tx,
            subscriptions: Vec::new(),
        };

        let mut connections = self.connections.write().await;
        connections.insert(agent_uuid, connection);

        Ok(Response::new(RegisterAgentResponse {
            success: true,
            error: None,
        }))
    }

    async fn subscribe(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<SubscribeResponse>, Status> {
        let req = request.into_inner();
        let agent_id = req
            .agent_id
            .ok_or_else(|| Status::invalid_argument("agent_id is required"))?;
        let agent_uuid = Uuid::parse_str(&agent_id.uuid)
            .map_err(|_| Status::invalid_argument("Invalid agent UUID"))?;

        // Update subscriptions
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions
            .entry(req.topic.clone())
            .or_insert_with(Vec::new)
            .push(agent_uuid);

        // Update agent's subscription list
        let mut connections = self.connections.write().await;
        if let Some(connection) = connections.get_mut(&agent_uuid) {
            connection.subscriptions.push(req.topic);
        }

        Ok(Response::new(SubscribeResponse {
            success: true,
            error: None,
        }))
    }

    async fn publish_message(
        &self,
        request: Request<PublishMessageRequest>,
    ) -> Result<Response<PublishMessageResponse>, Status> {
        let req = request.into_inner();

        // Send to internal channel for routing
        self.internal_tx
            .send(Event::PublishMessage {
                topic: req.topic,
                message: req.message,
            })
            .await
            .map_err(|_| Status::internal("Failed to publish message"))?;

        Ok(Response::new(PublishMessageResponse {
            success: true,
            error: None,
        }))
    }

    async fn send_message(
        &self,
        request: Request<SendMessageRequest>,
    ) -> Result<Response<SendMessageResponse>, Status> {
        let req = request.into_inner();
        let agent_id = req
            .agent_id
            .ok_or_else(|| Status::invalid_argument("agent_id is required"))?;
        let agent_uuid = Uuid::parse_str(&agent_id.uuid)
            .map_err(|_| Status::invalid_argument("Invalid agent UUID"))?;

        // Send to internal channel for routing
        self.internal_tx
            .send(Event::SendMessage {
                agent_id: agent_uuid,
                message: req.message,
            })
            .await
            .map_err(|_| Status::internal("Failed to send message"))?;

        Ok(Response::new(SendMessageResponse {
            success: true,
            error: None,
        }))
    }

    type StreamEventsStream = ReceiverStream<Result<ProtoEvent, Status>>;

    async fn stream_events(
        &self,
        request: Request<EventStreamRequest>,
    ) -> Result<Response<Self::StreamEventsStream>, Status> {
        let req = request.into_inner();

        let (tx, rx) = mpsc::channel(100);

        if let Some(agent_id) = req.agent_id {
            let agent_uuid = Uuid::parse_str(&agent_id.uuid)
                .map_err(|_| Status::invalid_argument("Invalid agent UUID"))?;

            // Update the connection's event sender
            let mut connections = self.connections.write().await;
            if let Some(connection) = connections.get_mut(&agent_uuid) {
                connection.event_tx = tx.clone();
            }
        }

        // Convert channel to stream with proper Result type
        let (result_tx, result_rx) = mpsc::channel(100);

        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                let _ = result_tx.send(Ok(event)).await;
            }
        });

        let stream = ReceiverStream::new(result_rx);

        Ok(Response::new(stream))
    }

    async fn send_event(&self, request: Request<ProtoEvent>) -> Result<Response<()>, Status> {
        let proto_event = request.into_inner();

        // Convert proto event to internal event
        let event = convert_proto_event_to_event(proto_event)
            .map_err(|e| Status::invalid_argument(format!("Invalid event: {}", e)))?;

        // Send to internal channel
        self.internal_tx
            .send(event)
            .await
            .map_err(|_| Status::internal("Failed to send event"))?;

        Ok(Response::new(()))
    }

    async fn heartbeat(
        &self,
        _request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        Ok(Response::new(HeartbeatResponse {}))
    }
}

/// Convert protobuf event to internal event
fn convert_proto_event_to_event(proto_event: ProtoEvent) -> Result<Event, Error> {
    use super::proto::event::EventType;

    match proto_event.event_type {
        Some(EventType::NewTask(e)) => {
            let agent_id = e
                .agent_id
                .ok_or_else(|| Error::RuntimeError(RuntimeError::EmptyTask))?;
            let agent_uuid = Uuid::parse_str(&agent_id.uuid)
                .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

            let task = e
                .task
                .ok_or_else(|| Error::RuntimeError(RuntimeError::EmptyTask))?;

            Ok(Event::NewTask {
                agent_id: agent_uuid,
                task: Task {
                    prompt: task.prompt,
                    submission_id: Uuid::parse_str(&task.submission_id)
                        .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?,
                    completed: task.completed,
                    result: task.result.map(|s| serde_json::Value::String(s)),
                    agent_id: task.agent_id.and_then(|id| Uuid::parse_str(&id.uuid).ok()),
                },
            })
        }
        Some(EventType::PublishMessage(e)) => Ok(Event::PublishMessage {
            topic: e.topic,
            message: e.message,
        }),
        Some(EventType::SendMessage(e)) => {
            let agent_id = e
                .agent_id
                .ok_or_else(|| Error::RuntimeError(RuntimeError::EmptyTask))?;
            let agent_uuid = Uuid::parse_str(&agent_id.uuid)
                .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

            Ok(Event::SendMessage {
                agent_id: agent_uuid,
                message: e.message,
            })
        }
        _ => Err(Error::RuntimeError(RuntimeError::EmptyTask)),
    }
}
