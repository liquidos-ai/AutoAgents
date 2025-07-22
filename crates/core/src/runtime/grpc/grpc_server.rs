use crate::{
    error::Error,
    protocol::AgentID,
    runtime::{Runtime, RuntimeError},
};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::{mpsc, RwLock};
use tonic::{transport::Server, Request, Response, Status};
use uuid::Uuid;

use super::proto::{
    agent_runtime_server::{AgentRuntime as AgentRuntimeService, AgentRuntimeServer},
    Event as ProtoEvent, EventStreamRequest, HeartbeatRequest, HeartbeatResponse,
    PublishMessageRequest, PublishMessageResponse, RegisterAgentRequest, RegisterAgentResponse,
    SendMessageRequest, SendMessageResponse, SubscribeRequest, SubscribeResponse,
};
use super::GrpcRuntime;

/// Standalone gRPC server for hosting a runtime
pub struct GrpcRuntimeServer {
    /// The runtime being served
    runtime: Arc<GrpcRuntime>,
    /// Server configuration
    config: GrpcServerConfig,
}

/// Configuration for the gRPC server
#[derive(Debug, Clone)]
pub struct GrpcServerConfig {
    /// Address to bind to
    pub bind_addr: SocketAddr,
    /// Maximum message size
    pub max_message_size: usize,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Enable reflection for debugging
    pub enable_reflection: bool,
}

impl Default for GrpcServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:50051".parse().unwrap(),
            max_message_size: 4 * 1024 * 1024, // 4MB
            worker_threads: 4,
            enable_reflection: false,
        }
    }
}

impl GrpcRuntimeServer {
    /// Create a new gRPC server with the given runtime
    pub fn new(runtime: Arc<GrpcRuntime>, config: GrpcServerConfig) -> Self {
        Self { runtime, config }
    }

    /// Start the server and run until shutdown
    pub async fn serve(self) -> Result<(), Error> {
        let addr = self.config.bind_addr;
        let max_message_size = self.config.max_message_size;

        // Create the service implementation
        let service = StandaloneGrpcService {
            runtime: self.runtime.clone(),
            connections: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        };

        let svc = AgentRuntimeServer::new(service)
            .max_decoding_message_size(max_message_size)
            .max_encoding_message_size(max_message_size);

        println!("gRPC server listening on {}", addr);

        Server::builder()
            .add_service(svc)
            .serve(addr)
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        Ok(())
    }
}

/// Standalone gRPC service implementation
struct StandaloneGrpcService {
    runtime: Arc<GrpcRuntime>,
    connections: Arc<RwLock<HashMap<AgentID, mpsc::Sender<ProtoEvent>>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<AgentID>>>>,
}

#[tonic::async_trait]
impl AgentRuntimeService for StandaloneGrpcService {
    async fn register_agent(
        &self,
        request: Request<RegisterAgentRequest>,
    ) -> Result<Response<RegisterAgentResponse>, Status> {
        let req = request.into_inner();
        let agent_id = req
            .agent_id
            .ok_or_else(|| Status::invalid_argument("agent_id is required"))?;
        let _agent_uuid = Uuid::parse_str(&agent_id.uuid)
            .map_err(|_| Status::invalid_argument("Invalid agent UUID"))?;

        // For standalone server, we just track the registration
        // The actual agent handling is done by the runtime

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

        // Update local tracking
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions
            .entry(req.topic)
            .or_insert_with(Vec::new)
            .push(agent_uuid);

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

        // Forward to runtime
        Runtime::publish_message(&*self.runtime, req.message, req.topic)
            .await
            .map_err(|e| Status::internal(format!("Failed to publish message: {}", e)))?;

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

        // Forward to runtime
        Runtime::send_message(&*self.runtime, req.message, agent_uuid)
            .await
            .map_err(|e| Status::internal(format!("Failed to send message: {}", e)))?;

        Ok(Response::new(SendMessageResponse {
            success: true,
            error: None,
        }))
    }

    type StreamEventsStream = tokio_stream::wrappers::ReceiverStream<Result<ProtoEvent, Status>>;

    async fn stream_events(
        &self,
        request: Request<EventStreamRequest>,
    ) -> Result<Response<Self::StreamEventsStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = mpsc::channel(100);

        if let Some(agent_id) = req.agent_id {
            let agent_uuid = Uuid::parse_str(&agent_id.uuid)
                .map_err(|_| Status::invalid_argument("Invalid agent UUID"))?;

            // Store the connection
            let mut connections = self.connections.write().await;
            connections.insert(agent_uuid, tx);
        }

        let (result_tx, result_rx) = mpsc::channel(100);

        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                let _ = result_tx.send(Ok(event)).await;
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(result_rx);
        Ok(Response::new(stream))
    }

    async fn send_event(&self, request: Request<ProtoEvent>) -> Result<Response<()>, Status> {
        let proto_event = request.into_inner();

        // Broadcast to all connected clients
        let connections = self.connections.read().await;
        for tx in connections.values() {
            let _ = tx.send(proto_event.clone()).await;
        }

        Ok(Response::new(()))
    }

    async fn heartbeat(
        &self,
        _request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        Ok(Response::new(HeartbeatResponse {}))
    }
}

/// Builder for creating a gRPC server
pub struct GrpcServerBuilder {
    config: GrpcServerConfig,
}

impl GrpcServerBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: GrpcServerConfig::default(),
        }
    }

    /// Set the bind address
    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.config.bind_addr = addr;
        self
    }

    /// Set the maximum message size
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.config.max_message_size = size;
        self
    }

    /// Set the number of worker threads
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.worker_threads = threads;
        self
    }

    /// Enable reflection for debugging
    pub fn enable_reflection(mut self, enable: bool) -> Self {
        self.config.enable_reflection = enable;
        self
    }

    /// Build and return the server
    pub fn build(self, runtime: Arc<GrpcRuntime>) -> GrpcRuntimeServer {
        GrpcRuntimeServer::new(runtime, self.config)
    }
}

impl Default for GrpcServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
