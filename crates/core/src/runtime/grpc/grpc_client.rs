#![allow(dead_code)]
use crate::{
    error::Error,
    protocol::{AgentID, Event, TaskResult},
    runtime::{RuntimeError, Task},
};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tonic::{transport::Channel, Request};
use uuid::Uuid;

use super::proto::{
    agent_runtime_client::AgentRuntimeClient, Event as ProtoEvent, EventStreamRequest,
    PublishMessageRequest, RegisterAgentRequest, SendMessageRequest, SubscribeRequest,
};

/// Configuration for gRPC client
#[derive(Debug, Clone)]
pub struct GrpcClientConfig {
    /// Server address to connect to
    pub server_addr: String,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    /// Maximum retry attempts
    pub max_retries: usize,
}

impl Default for GrpcClientConfig {
    fn default() -> Self {
        Self {
            server_addr: "http://127.0.0.1:50051".to_string(),
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(10),
            max_retries: 3,
        }
    }
}

/// gRPC client for connecting to a distributed runtime
pub struct GrpcRuntimeClient {
    /// Agent ID
    agent_id: AgentID,
    /// Agent name
    agent_name: String,
    /// gRPC client
    client: AgentRuntimeClient<Channel>,
    /// Event receiver
    event_rx: Option<mpsc::Receiver<Event>>,
    /// Shutdown signal
    shutdown_tx: Option<mpsc::Sender<()>>,
    /// Configuration
    config: GrpcClientConfig,
}

impl GrpcRuntimeClient {
    /// Connect to a gRPC runtime server
    pub async fn connect(
        config: GrpcClientConfig,
        agent_id: AgentID,
        agent_name: String,
    ) -> Result<Self, Error> {
        // Connect to server
        let channel = Channel::from_shared(config.server_addr.clone())
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .connect()
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        let mut client = AgentRuntimeClient::new(channel);

        // Register agent
        let request = Request::new(RegisterAgentRequest {
            agent_id: Some(super::proto::AgentId {
                uuid: agent_id.to_string(),
            }),
            agent_name: agent_name.clone(),
            description: String::new(),
        });

        let response = client
            .register_agent(request)
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        if !response.into_inner().success {
            return Err(Error::RuntimeError(RuntimeError::EmptyTask));
        }

        // Start event stream
        let (event_tx, event_rx) = mpsc::channel(100);
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);

        let mut stream_client = client.clone();
        let agent_id_clone = agent_id;

        tokio::spawn(async move {
            let request = Request::new(EventStreamRequest {
                agent_id: Some(super::proto::AgentId {
                    uuid: agent_id_clone.to_string(),
                }),
            });

            match stream_client.stream_events(request).await {
                Ok(response) => {
                    let mut stream = response.into_inner();
                    loop {
                        tokio::select! {
                            event = stream.next() => {
                                match event {
                                    Some(Ok(proto_event)) => {
                                        if let Ok(event) = convert_proto_event_to_event(proto_event) {
                                            let _ = event_tx.send(event).await;
                                        }
                                    }
                                    Some(Err(e)) => {
                                        log::error!("Stream error: {}", e);
                                        break;
                                    }
                                    None => {
                                        log::info!("Event stream closed");
                                        break;
                                    }
                                }
                            }
                            _ = shutdown_rx.recv() => {
                                log::info!("Shutting down event stream");
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to start event stream: {}", e);
                }
            }
        });

        Ok(Self {
            agent_id,
            agent_name,
            client,
            event_rx: Some(event_rx),
            shutdown_tx: Some(shutdown_tx),
            config,
        })
    }

    /// Subscribe to a topic
    pub async fn subscribe(&mut self, topic: String) -> Result<(), Error> {
        let request = Request::new(SubscribeRequest {
            agent_id: Some(super::proto::AgentId {
                uuid: self.agent_id.to_string(),
            }),
            topic,
        });

        let response = self
            .client
            .subscribe(request)
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        if !response.into_inner().success {
            return Err(Error::RuntimeError(RuntimeError::EmptyTask));
        }

        Ok(())
    }

    /// Send an event to the runtime
    pub async fn send_event(&mut self, event: Event) -> Result<(), Error> {
        let proto_event = convert_event_to_proto(event)?;
        let request = Request::new(proto_event);

        self.client
            .send_event(request)
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        Ok(())
    }

    /// Publish a message to a topic
    pub async fn publish_message(&mut self, topic: String, message: String) -> Result<(), Error> {
        let request = Request::new(PublishMessageRequest { topic, message });

        let response = self
            .client
            .publish_message(request)
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        if !response.into_inner().success {
            return Err(Error::RuntimeError(RuntimeError::EmptyTask));
        }

        Ok(())
    }

    /// Send a message directly to an agent
    pub async fn send_message(&mut self, agent_id: AgentID, message: String) -> Result<(), Error> {
        let request = Request::new(SendMessageRequest {
            agent_id: Some(super::proto::AgentId {
                uuid: agent_id.to_string(),
            }),
            message,
        });

        let response = self
            .client
            .send_message(request)
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        if !response.into_inner().success {
            return Err(Error::RuntimeError(RuntimeError::EmptyTask));
        }

        Ok(())
    }

    /// Take the event receiver
    pub async fn take_event_receiver(&mut self) -> Option<mpsc::Receiver<Event>> {
        self.event_rx.take()
    }

    /// Get agent ID
    pub fn agent_id(&self) -> AgentID {
        self.agent_id
    }

    /// Get agent name
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// Run the client (keeps connection alive)
    pub async fn run(self) -> Result<(), Error> {
        // Keep the client alive
        tokio::signal::ctrl_c()
            .await
            .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?;

        Ok(())
    }

    /// Shutdown the client
    pub async fn shutdown(self) -> Result<(), Error> {
        if let Some(shutdown_tx) = self.shutdown_tx {
            let _ = shutdown_tx.send(()).await;
        }
        Ok(())
    }
}

/// Convert internal event to protobuf event
fn convert_event_to_proto(event: Event) -> Result<ProtoEvent, Error> {
    use super::proto::event::EventType;

    let event_type = match event {
        Event::NewTask { agent_id, task } => Some(EventType::NewTask(super::proto::NewTaskEvent {
            agent_id: Some(super::proto::AgentId {
                uuid: agent_id.to_string(),
            }),
            task: Some(super::proto::Task {
                prompt: task.prompt,
                submission_id: task.submission_id.to_string(),
                completed: task.completed,
                result: task.result.map(|v| v.to_string()),
                agent_id: None,
            }),
        })),
        Event::TaskStarted {
            sub_id,
            agent_id,
            task_description,
        } => Some(EventType::TaskStarted(super::proto::TaskStartedEvent {
            submission_id: sub_id.to_string(),
            agent_id: Some(super::proto::AgentId {
                uuid: agent_id.to_string(),
            }),
            task_description,
        })),
        Event::TaskComplete { sub_id, result } => {
            Some(EventType::TaskComplete(super::proto::TaskCompleteEvent {
                submission_id: sub_id.to_string(),
                result: Some(match result {
                    TaskResult::Value(v) => super::proto::TaskResult {
                        result: Some(super::proto::task_result::Result::Value(
                            serde_json::to_string(&v).unwrap_or_default(),
                        )),
                    },
                    TaskResult::Failure(e) => super::proto::TaskResult {
                        result: Some(super::proto::task_result::Result::Error(e)),
                    },
                    TaskResult::Aborted => super::proto::TaskResult {
                        result: Some(super::proto::task_result::Result::Error(
                            "Aborted".to_string(),
                        )),
                    },
                }),
            }))
        }
        Event::TaskError { sub_id, result } => {
            Some(EventType::TaskComplete(super::proto::TaskCompleteEvent {
                submission_id: sub_id.to_string(),
                result: Some(match result {
                    TaskResult::Value(_) => super::proto::TaskResult {
                        result: Some(super::proto::task_result::Result::Error(
                            "Unexpected value in error".to_string(),
                        )),
                    },
                    TaskResult::Failure(e) => super::proto::TaskResult {
                        result: Some(super::proto::task_result::Result::Error(e)),
                    },
                    TaskResult::Aborted => super::proto::TaskResult {
                        result: Some(super::proto::task_result::Result::Error(
                            "Aborted".to_string(),
                        )),
                    },
                }),
            }))
        }
        Event::PublishMessage { topic, message } => Some(EventType::PublishMessage(
            super::proto::PublishMessageEvent { topic, message },
        )),
        Event::SendMessage { agent_id, message } => {
            Some(EventType::SendMessage(super::proto::SendMessageEvent {
                agent_id: Some(super::proto::AgentId {
                    uuid: agent_id.to_string(),
                }),
                message,
            }))
        }
        _ => None,
    };

    Ok(ProtoEvent { event_type })
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
        Some(EventType::TaskStarted(e)) => Ok(Event::TaskStarted {
            sub_id: Uuid::parse_str(&e.submission_id)
                .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?,
            agent_id: e
                .agent_id
                .and_then(|id| Uuid::parse_str(&id.uuid).ok())
                .ok_or_else(|| Error::RuntimeError(RuntimeError::EmptyTask))?,
            task_description: e.task_description,
        }),
        Some(EventType::TaskComplete(e)) => {
            let result = e
                .result
                .ok_or_else(|| Error::RuntimeError(RuntimeError::EmptyTask))?;

            let task_result = match result.result {
                Some(super::proto::task_result::Result::Value(v)) => {
                    TaskResult::Value(serde_json::from_str(&v).unwrap_or_default())
                }
                Some(super::proto::task_result::Result::Error(e)) => TaskResult::Failure(e),
                None => return Err(Error::RuntimeError(RuntimeError::EmptyTask)),
            };

            Ok(Event::TaskComplete {
                sub_id: Uuid::parse_str(&e.submission_id)
                    .map_err(|_| Error::RuntimeError(RuntimeError::EmptyTask))?,
                result: task_result,
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
