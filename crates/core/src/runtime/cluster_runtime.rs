use async_trait::async_trait;
use log::{debug, error, info, warn};
use ractor::{Actor, ActorRef};
use ractor_cluster::node::{NodeConnectionMode, NodeServer, NodeServerMessage};
use ractor_cluster::IncomingEncryptionMode;
use serde::{Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::{
    actor::{AnyActor, Transport},
    agent::task::Task,
    protocol::{Event, InternalEvent, RuntimeID},
};

use super::{Runtime, RuntimeError};

const DEFAULT_CHANNEL_BUFFER: usize = 100;
const DEFAULT_INTERNAL_BUFFER: usize = 1000;

/// Cluster-aware message wrapper for distributing tasks across nodes
#[cfg(feature = "cluster")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMessage {
    Task {
        task: Task,
        source_agent_id: Option<String>,
        target_topic: String,
    },
    SubscriptionRegistration {
        client_id: String,
        topic_name: String,
        subscribe: bool, // true = subscribe, false = unsubscribe
    },
}

// Manual BytesConvertable implementation for cluster mode since the derive isn't working
#[cfg(feature = "cluster")]
impl ractor::BytesConvertable for ClusterMessage {
    fn into_bytes(self) -> Vec<u8> {
        serde_json::to_vec(&self).expect("Failed to serialize ClusterMessage")
    }

    fn from_bytes(data: Vec<u8>) -> Self {
        serde_json::from_slice(&data).expect("Failed to deserialize ClusterMessage")
    }
}

#[cfg(feature = "cluster")]
impl crate::actor::ActorMessage for ClusterMessage {}

#[cfg(feature = "cluster")]
impl crate::actor::CloneableMessage for ClusterMessage {}

impl From<Task> for ClusterMessage {
    fn from(task: Task) -> Self {
        ClusterMessage::Task {
            task,
            source_agent_id: None,
            target_topic: "default".to_string(),
        }
    }
}

impl From<ClusterMessage> for Task {
    fn from(msg: ClusterMessage) -> Self {
        match msg {
            ClusterMessage::Task { task, .. } => task,
            ClusterMessage::SubscriptionRegistration { .. } => {
                panic!("Cannot convert SubscriptionRegistration to Task")
            }
        }
    }
}

/// Topic subscription entry storing type information and actor references
#[derive(Debug)]
struct Subscription {
    topic_type: TypeId,
    actors: Vec<Arc<dyn AnyActor>>,
    // Store raw actor cells for process group operations
    actor_cells: Vec<ractor::ActorCell>,
}

/// Host runtime for cluster coordination - manages all client connections and routes events
#[derive(Debug, Clone)]
pub struct ClusterHostRuntime {
    pub id: RuntimeID,
    // External event channel for application consumption
    external_tx: mpsc::Sender<Event>,
    external_rx: Arc<Mutex<Option<mpsc::Receiver<Event>>>>,
    // Internal event channel for runtime processing
    internal_tx: mpsc::Sender<InternalEvent>,
    internal_rx: Arc<Mutex<Option<mpsc::Receiver<InternalEvent>>>>,
    // Global subscriptions map: topic_name -> Subscription (across all clients)
    global_subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
    // Client tracking: node_id -> client info
    connected_clients: Arc<RwLock<HashMap<String, ClientInfo>>>,
    // Client subscriptions mapping: topic_name -> set of client_ids that have subscriptions
    client_subscriptions: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    // Transport layer for message delivery
    transport: Arc<dyn Transport>,
    // Node server reference
    node_ref: Arc<Mutex<Option<ActorRef<NodeServerMessage>>>>,
    // Runtime state
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    // Cluster configuration
    node_name: String,
    cookie: String,
    port: u16,
    host: String,
}

/// Client runtime that connects to a ClusterHostRuntime
#[derive(Debug, Clone)]
pub struct ClusterClientRuntime {
    pub id: RuntimeID,
    // External event channel for application consumption
    external_tx: mpsc::Sender<Event>,
    external_rx: Arc<Mutex<Option<mpsc::Receiver<Event>>>>,
    // Internal event channel for runtime processing
    internal_tx: mpsc::Sender<InternalEvent>,
    internal_rx: Arc<Mutex<Option<mpsc::Receiver<InternalEvent>>>>,
    // Local subscriptions map: topic_name -> Subscription
    local_subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
    // Transport layer for message delivery
    transport: Arc<dyn Transport>,
    // Node server reference
    node_ref: Arc<Mutex<Option<ActorRef<NodeServerMessage>>>>,
    // Runtime state
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    // Client configuration
    client_id: String,
    host_address: String,
    node_name: String,
    cookie: String,
    port: u16,
    host: String,
}

/// Information about connected clients
#[derive(Debug, Clone)]
struct ClientInfo {
    client_id: String,
    node_id: String,
    subscriptions: Vec<String>, // Topics this client is subscribed to
}

/// Legacy cluster runtime - deprecated, use ClusterHostRuntime or ClusterClientRuntime
#[derive(Debug, Clone)]
pub struct ClusterRuntime {
    pub id: RuntimeID,
    // External event channel for application consumption
    external_tx: mpsc::Sender<Event>,
    external_rx: Arc<Mutex<Option<mpsc::Receiver<Event>>>>,
    // Internal event channel for runtime processing
    internal_tx: mpsc::Sender<InternalEvent>,
    internal_rx: Arc<Mutex<Option<mpsc::Receiver<InternalEvent>>>>,
    // Subscriptions map: topic_name -> Subscription
    subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
    // Transport layer for message delivery
    transport: Arc<dyn Transport>,
    // Node server reference
    node_ref: Arc<Mutex<Option<ActorRef<NodeServerMessage>>>>,
    // Runtime state
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    // Cluster configuration
    node_name: String,
    cookie: String,
    port: u16,
    host: String,
}

impl ClusterHostRuntime {
    pub fn new(node_name: String, cookie: String, port: u16, host: String) -> Arc<Self> {
        Self::with_transport(
            node_name,
            cookie,
            port,
            host,
            Arc::new(crate::actor::LocalTransport),
        )
    }

    pub fn with_transport(
        node_name: String,
        cookie: String,
        port: u16,
        host: String,
        transport: Arc<dyn Transport>,
    ) -> Arc<Self> {
        let id = Uuid::new_v4();
        let buffer_size = DEFAULT_CHANNEL_BUFFER;

        // Create channels
        let (external_tx, external_rx) = mpsc::channel(buffer_size);
        let (internal_tx, internal_rx) = mpsc::channel(DEFAULT_INTERNAL_BUFFER);

        Arc::new(Self {
            id,
            external_tx,
            external_rx: Arc::new(Mutex::new(Some(external_rx))),
            internal_tx,
            internal_rx: Arc::new(Mutex::new(Some(internal_rx))),
            global_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            connected_clients: Arc::new(RwLock::new(HashMap::new())),
            client_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            transport,
            node_ref: Arc::new(Mutex::new(None)),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
            node_name,
            cookie,
            port,
            host,
        })
    }

    /// Register a new client connection
    async fn register_client(&self, client_info: ClientInfo) {
        let mut clients = self.connected_clients.write().await;
        info!("Registering new client: {}", client_info.client_id);
        clients.insert(client_info.node_id.clone(), client_info);
    }

    /// Unregister a client connection
    async fn unregister_client(&self, node_id: &str) {
        let mut clients = self.connected_clients.write().await;
        if let Some(client) = clients.remove(node_id) {
            info!("Unregistering client: {}", client.client_id);
            // Clean up client subscriptions
            self.cleanup_client_subscriptions(&client).await;
        }
    }

    /// Clean up subscriptions for a disconnected client
    async fn cleanup_client_subscriptions(&self, client: &ClientInfo) {
        let mut subscriptions = self.global_subscriptions.write().await;
        for topic in &client.subscriptions {
            if let Some(subscription) = subscriptions.get_mut(topic) {
                // Remove actors belonging to this client (simplified - in real implementation
                // we'd need to track which actors belong to which client)
                subscription.actors.retain(|_| true); // TODO: Implement proper client-actor tracking
            }
        }
    }

    /// Handle message publishing to global topic subscribers across all clients
    async fn handle_publish_message_global(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        info!("Handling global publish event: {topic_name}");

        let subscriptions = self.global_subscriptions.read().await;

        // Send to any local subscribers on the host (usually none)
        if let Some(subscription) = subscriptions.get(topic_name) {
            // Verify type safety
            if subscription.topic_type != topic_type {
                error!(
                    "Type mismatch for topic '{}': expected {:?}, got {:?}",
                    topic_name, subscription.topic_type, topic_type
                );
                return Err(RuntimeError::TopicTypeMismatch(
                    topic_name.to_owned(),
                    topic_type,
                ));
            }

            // Send to all subscribed actors across all clients (on host itself - usually none)
            for actor in &subscription.actors {
                if let Err(e) = self
                    .transport
                    .send(actor.as_ref(), Arc::clone(&message))
                    .await
                {
                    error!("Failed to send message to local host subscriber: {e}");
                }
            }

            info!(
                "Message sent to {} local host subscribers for topic: {}",
                subscription.actors.len(),
                topic_name
            );
        } else {
            info!("No local host subscribers for topic: {}", topic_name);
        }

        // Always distribute to remote clients via cluster communication - this is the main distribution path
        info!(
            "Distributing message to remote clients for topic: {}",
            topic_name
        );
        self.distribute_to_clients(topic_name, message).await;

        Ok(())
    }

    /// Distribute message to all connected clients
    async fn distribute_to_clients(&self, topic_name: &str, message: Arc<dyn Any + Send + Sync>) {
        // Check which clients have subscriptions for this specific topic using our tracking
        let client_subs = self.client_subscriptions.read().await;
        let subscribers_for_topic = client_subs.get(topic_name);

        if let Some(subscribers) = subscribers_for_topic {
            if subscribers.is_empty() {
                info!(
                    "No client subscribers for topic '{}', skipping distribution",
                    topic_name
                );
                return;
            }
            info!(
                "Found {} client subscribers for topic '{}': {:?}",
                subscribers.len(),
                topic_name,
                subscribers
            );
        } else {
            info!(
                "No client subscriptions found for topic '{}', skipping distribution",
                topic_name
            );
            return;
        }

        // Get all cluster communication actors
        let all_comm_actors = ractor::pg::get_members(&"cluster_communication".to_string());

        // Filter for remote actors (client forwarders)
        let client_actors: Vec<_> = all_comm_actors
            .into_iter()
            .filter(|actor| !actor.get_id().is_local()) // Only remote (client) forwarders
            .collect();

        info!(
            "Found {} cluster communication actors, {} are client forwarders for topic: {}",
            ractor::pg::get_members(&"cluster_communication".to_string()).len(),
            client_actors.len(),
            topic_name
        );

        if !client_actors.is_empty() {
            info!(
                "Distributing message to {} connected clients for topic: {}",
                client_actors.len(),
                topic_name
            );

            // Convert message to ClusterMessage and send to client forwarders
            if let Some(task) = message.downcast_ref::<Task>() {
                let cluster_msg = ClusterMessage::Task {
                    task: task.clone(),
                    source_agent_id: None, // Host doesn't have a specific agent_id
                    target_topic: topic_name.to_string(),
                };
                info!(
                    "Converting task to ClusterMessage for distribution: {}",
                    task.prompt.chars().take(50).collect::<String>()
                );

                for (i, client_actor) in client_actors.iter().enumerate() {
                    info!(
                        "Sending to client forwarder {} of {}: {:?}",
                        i + 1,
                        client_actors.len(),
                        client_actor.get_id()
                    );

                    let forwarder_ref = ActorRef::<ClusterMessage>::from(client_actor.clone());
                    if let Err(e) = forwarder_ref.cast(cluster_msg.clone()) {
                        error!(
                            "Failed to send cluster message to client {:?}: {}",
                            client_actor.get_id(),
                            e
                        );
                    } else {
                        info!(
                            "Successfully sent cluster message to client: {:?}",
                            client_actor.get_id()
                        );
                    }
                }
            } else {
                warn!("Message could not be converted to Task for cluster distribution");
            }
        } else {
            warn!("No connected clients found for distribution - messages will only reach local subscribers");
            // Log all process group members for debugging
            let all_members = ractor::pg::get_members(&"cluster_communication".to_string());
            for member in all_members {
                info!(
                    "Process group member: {:?} (local: {})",
                    member.get_id(),
                    member.get_id().is_local()
                );
            }
        }
    }

    /// Handle actor subscription to a topic (from any client)
    async fn handle_global_subscribe(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
        actor_cell: Option<ractor::ActorCell>,
        client_id: Option<String>,
    ) -> Result<(), RuntimeError> {
        info!("Global subscription to topic: {topic_name}");

        let mut subscriptions = self.global_subscriptions.write().await;

        match subscriptions.get_mut(topic_name) {
            Some(subscription) => {
                // Verify type consistency
                if subscription.topic_type != topic_type {
                    return Err(RuntimeError::TopicTypeMismatch(
                        topic_name.to_string(),
                        subscription.topic_type,
                    ));
                }
                subscription.actors.push(actor.clone());
                if let Some(ref cell) = actor_cell {
                    subscription.actor_cells.push(cell.clone());
                }
            }
            None => {
                // Create new subscription
                let mut actor_cells = Vec::new();
                if let Some(ref cell) = actor_cell {
                    actor_cells.push(cell.clone());
                }

                subscriptions.insert(
                    topic_name.to_string(),
                    Subscription {
                        topic_type,
                        actors: vec![actor.clone()],
                        actor_cells,
                    },
                );
            }
        }

        // Update client subscription tracking if client_id is provided
        if let Some(client_id) = client_id {
            self.update_client_subscription(&client_id, topic_name)
                .await;
        }

        // Join the cluster-wide process group for this topic if we have an actor cell
        if let Some(cell) = actor_cell {
            self.join_process_group(topic_name, cell).await;
        }

        Ok(())
    }

    /// Update client subscription tracking
    async fn update_client_subscription(&self, client_id: &str, topic_name: &str) {
        let mut clients = self.connected_clients.write().await;
        for (_, client_info) in clients.iter_mut() {
            if client_info.client_id == client_id {
                if !client_info.subscriptions.contains(&topic_name.to_string()) {
                    client_info.subscriptions.push(topic_name.to_string());
                }
                break;
            }
        }
    }

    /// Handle subscription registration from clients
    async fn handle_subscription_registration(
        &self,
        client_id: String,
        topic_name: String,
        subscribe: bool,
    ) {
        let mut client_subs = self.client_subscriptions.write().await;

        if subscribe {
            // Add client to topic subscribers
            client_subs
                .entry(topic_name.clone())
                .or_insert_with(HashSet::new)
                .insert(client_id.clone());
            info!(
                "Client '{}' subscribed to topic '{}'. Total subscribers for topic: {}",
                client_id,
                topic_name,
                client_subs.get(&topic_name).map(|s| s.len()).unwrap_or(0)
            );
        } else {
            // Remove client from topic subscribers
            if let Some(subscribers) = client_subs.get_mut(&topic_name) {
                subscribers.remove(&client_id);
                if subscribers.is_empty() {
                    client_subs.remove(&topic_name);
                }
            }
            info!(
                "Client '{}' unsubscribed from topic '{}'. Remaining subscribers: {}",
                client_id,
                topic_name,
                client_subs.get(&topic_name).map(|s| s.len()).unwrap_or(0)
            );
        }
    }

    /// Join a process group for cluster-wide communication
    async fn join_process_group(&self, topic_name: &str, actor_cell: ractor::ActorCell) {
        let group_name = format!("topic_{}", topic_name);

        info!(
            "Joining process group '{}' for cluster-wide communication",
            group_name
        );

        ractor::pg::join(group_name.clone(), vec![actor_cell]);
        info!("Successfully joined process group '{}'", group_name);
    }

    /// Process internal events in the runtime
    async fn process_internal_event(
        &self,
        event: InternalEvent,
    ) -> Result<(), crate::error::Error> {
        debug!("Received internal event: {event:?}");
        match event {
            InternalEvent::ProtocolEvent(event) => {
                self.process_protocol_event(event).await?;
            }
            InternalEvent::Shutdown => {
                self.shutdown_flag.store(true, Ordering::SeqCst);
                self.shutdown_notify.notify_waiters();
            }
        }
        Ok(())
    }

    /// Forward protocol events to external channel
    async fn process_protocol_event(&self, event: Event) -> Result<(), crate::error::Error> {
        match event {
            Event::PublishMessage {
                topic_type,
                topic_name,
                message,
            } => {
                self.handle_publish_message_global(&topic_name, topic_type, message)
                    .await?;
            }
            _ => {
                // Other protocol events are sent to external
                self.external_tx
                    .send(event)
                    .await
                    .map_err(RuntimeError::EventError)?;
            }
        }
        Ok(())
    }

    /// Start the internal event processing loop
    async fn event_loop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut internal_rx = self
            .internal_rx
            .lock()
            .await
            .take()
            .ok_or("Internal receiver already taken")?;

        info!("ClusterHostRuntime event loop starting");

        loop {
            tokio::select! {
                Some(event) = internal_rx.recv() => {
                    debug!("Processing internal event");

                    if matches!(event, InternalEvent::Shutdown) {
                        info!("Received shutdown event");
                        self.process_internal_event(event).await?;
                        break;
                    }

                    if let Err(e) = self.process_internal_event(event).await {
                        error!("Error processing internal event: {e}");
                        break;
                    }
                }
                _ = self.shutdown_notify.notified() => {
                    if self.shutdown_flag.load(Ordering::SeqCst) {
                        info!("Runtime received shutdown notification");
                        break;
                    }
                }
                else => {
                    warn!("Internal event channel closed");
                    break;
                }
            }
        }

        // Drain remaining events
        info!("Draining remaining events before shutdown");
        while let Ok(event) = internal_rx.try_recv() {
            if let Err(e) = self.process_internal_event(event).await {
                error!("Error processing event during shutdown: {e}");
            }
        }

        info!("ClusterHostRuntime event loop stopped");
        Ok(())
    }

    /// Initialize the node server for cluster communication
    async fn init_node_server(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let node = NodeServer::new(
            self.port,
            self.cookie.clone(),
            format!("{}@{}:{}", self.node_name, self.host, self.port),
            self.host.clone(),
            Some(IncomingEncryptionMode::Raw),
            Some(NodeConnectionMode::Isolated),
        );

        let (node_ref, _node_handle) = Actor::spawn(None, node, ())
            .await
            .map_err(|e| format!("Failed to start node server: {}", e))?;

        *self.node_ref.lock().await = Some(node_ref);
        info!(
            "ClusterHostRuntime node server initialized on {}:{}",
            self.host, self.port
        );

        Ok(())
    }

    /// Initialize cluster communication forwarder for host
    async fn init_cluster_communication(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing cluster communication forwarder for host");

        let forwarder = ClusterHostForwarder {
            runtime: Arc::new(self.clone()),
        };

        let (forwarder_ref, _handle) = ractor::Actor::spawn(None, forwarder, ())
            .await
            .map_err(|e| format!("Failed to start cluster forwarder: {}", e))?;

        ractor::pg::join(
            "cluster_communication".to_string(),
            vec![forwarder_ref.get_cell()],
        );

        info!("✅ ClusterHostRuntime communication forwarder initialized and joined process group");
        Ok(())
    }
}

impl ClusterClientRuntime {
    pub fn new(
        client_id: String,
        host_address: String,
        node_name: String,
        cookie: String,
        port: u16,
        host: String,
    ) -> Arc<Self> {
        Self::with_transport(
            client_id,
            host_address,
            node_name,
            cookie,
            port,
            host,
            Arc::new(crate::actor::LocalTransport),
        )
    }

    pub fn with_transport(
        client_id: String,
        host_address: String,
        node_name: String,
        cookie: String,
        port: u16,
        host: String,
        transport: Arc<dyn Transport>,
    ) -> Arc<Self> {
        let id = Uuid::new_v4();
        let buffer_size = DEFAULT_CHANNEL_BUFFER;

        // Create channels
        let (external_tx, external_rx) = mpsc::channel(buffer_size);
        let (internal_tx, internal_rx) = mpsc::channel(DEFAULT_INTERNAL_BUFFER);

        Arc::new(Self {
            id,
            external_tx,
            external_rx: Arc::new(Mutex::new(Some(external_rx))),
            internal_tx,
            internal_rx: Arc::new(Mutex::new(Some(internal_rx))),
            local_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            transport,
            node_ref: Arc::new(Mutex::new(None)),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
            client_id,
            host_address,
            node_name,
            cookie,
            port,
            host,
        })
    }

    /// Connect to the cluster host
    pub async fn connect_to_host(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let node_ref = self.node_ref.lock().await;
        if let Some(ref node) = *node_ref {
            info!(
                "Attempting to connect to cluster host: {} (client: {})",
                self.host_address, self.client_id
            );

            match ractor_cluster::client_connect(node, &self.host_address).await {
                Ok(_) => {
                    info!(
                        "✅ Successfully connected to cluster host: {} (client: {})",
                        self.host_address, self.client_id
                    );
                }
                Err(e) => {
                    error!(
                        "❌ Failed to connect to cluster host {}: {} (client: {})",
                        self.host_address, e, self.client_id
                    );
                    return Err(format!("Cluster connection failed: {}", e).into());
                }
            }

            // Give some time for the connection to stabilize
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Initialize cluster communication after connection is established
            match self.init_cluster_communication().await {
                Ok(_) => {
                    info!(
                        "✅ Cluster communication initialized for client: {}",
                        self.client_id
                    );
                }
                Err(e) => {
                    error!(
                        "❌ Failed to initialize cluster communication for client {}: {}",
                        self.client_id, e
                    );
                    return Err(e);
                }
            }

            // Send all existing subscriptions to the host now that we're connected
            self.register_existing_subscriptions_with_host().await;
        } else {
            return Err("Node server not started".into());
        }
        Ok(())
    }

    /// Handle local message publishing - forwards to host for routing
    async fn handle_publish_message_local(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        debug!("Handling local publish event: {topic_name}");

        // First, send to local subscribers
        self.deliver_to_local_subscribers_only(topic_name, topic_type, Arc::clone(&message))
            .await?;

        // Forward to host for global routing
        self.forward_to_host(topic_name, message).await;

        Ok(())
    }

    /// Deliver message only to local subscribers without forwarding to host
    async fn deliver_to_local_subscribers_only(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        debug!("Delivering to local subscribers only for topic: {topic_name}");

        let subscriptions = self.local_subscriptions.read().await;
        if let Some(subscription) = subscriptions.get(topic_name) {
            // Verify type safety
            if subscription.topic_type != topic_type {
                error!(
                    "Type mismatch for topic '{}': expected {:?}, got {:?}",
                    topic_name, subscription.topic_type, topic_type
                );
                return Err(RuntimeError::TopicTypeMismatch(
                    topic_name.to_owned(),
                    topic_type,
                ));
            }

            // Send to all local subscribed actors
            for actor in &subscription.actors {
                if let Err(e) = self
                    .transport
                    .send(actor.as_ref(), Arc::clone(&message))
                    .await
                {
                    error!("Failed to send message to local subscriber: {e}");
                }
            }

            info!(
                "Message delivered to {} local subscribers for topic: {}",
                subscription.actors.len(),
                topic_name
            );
        } else {
            debug!("No local subscribers found for topic: {}", topic_name);
        }

        Ok(())
    }

    /// Forward message to host for global routing
    async fn forward_to_host(&self, topic_name: &str, message: Arc<dyn Any + Send + Sync>) {
        info!(
            "Forwarding message for topic '{}' to host for global routing",
            topic_name
        );

        // Get all cluster communication actors (including host)
        let all_comm_actors = ractor::pg::get_members(&"cluster_communication".to_string());

        // Filter for remote actors (host forwarders)
        let host_forwarders: Vec<_> = all_comm_actors
            .into_iter()
            .filter(|actor| !actor.get_id().is_local()) // Only remote (host) forwarders
            .collect();

        info!(
            "Found {} cluster communication actors, {} are host forwarders for topic: {}",
            ractor::pg::get_members(&"cluster_communication".to_string()).len(),
            host_forwarders.len(),
            topic_name
        );

        if !host_forwarders.is_empty() {
            info!(
                "Forwarding message to {} host forwarders for topic: {}",
                host_forwarders.len(),
                topic_name
            );

            // Convert message to ClusterMessage and send to host forwarders
            if let Some(task) = message.downcast_ref::<Task>() {
                let cluster_msg = ClusterMessage::Task {
                    task: task.clone(),
                    source_agent_id: Some(self.client_id.clone()),
                    target_topic: topic_name.to_string(),
                };

                for host_forwarder in host_forwarders {
                    let forwarder_ref = ActorRef::<ClusterMessage>::from(host_forwarder.clone());
                    if let Err(e) = forwarder_ref.cast(cluster_msg.clone()) {
                        error!(
                            "Failed to send cluster message to host forwarder {:?}: {}",
                            host_forwarder.get_id(),
                            e
                        );
                    } else {
                        info!(
                            "Successfully sent cluster message to host forwarder: {:?}",
                            host_forwarder.get_id()
                        );
                    }
                }
            } else {
                warn!("Message could not be converted to Task for cluster forwarding");
            }
        } else {
            warn!("No host forwarders found for message forwarding - messages will only reach local subscribers");
            // Log all process group members for debugging
            let all_members = ractor::pg::get_members(&"cluster_communication".to_string());
            for member in all_members {
                info!(
                    "Process group member: {:?} (local: {})",
                    member.get_id(),
                    member.get_id().is_local()
                );
            }
        }
    }

    /// Handle actor subscription to a local topic and notify host
    async fn handle_local_subscribe(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
        actor_cell: Option<ractor::ActorCell>,
    ) -> Result<(), RuntimeError> {
        info!("Local subscription to topic: {topic_name}");

        let is_new_topic = {
            let subscriptions = self.local_subscriptions.read().await;
            !subscriptions.contains_key(topic_name)
        };

        let mut subscriptions = self.local_subscriptions.write().await;

        match subscriptions.get_mut(topic_name) {
            Some(subscription) => {
                // Verify type consistency
                if subscription.topic_type != topic_type {
                    return Err(RuntimeError::TopicTypeMismatch(
                        topic_name.to_string(),
                        subscription.topic_type,
                    ));
                }
                subscription.actors.push(actor.clone());
                if let Some(ref cell) = actor_cell {
                    subscription.actor_cells.push(cell.clone());
                }
            }
            None => {
                // Create new subscription
                let mut actor_cells = Vec::new();
                if let Some(ref cell) = actor_cell {
                    actor_cells.push(cell.clone());
                }

                subscriptions.insert(
                    topic_name.to_string(),
                    Subscription {
                        topic_type,
                        actors: vec![actor.clone()],
                        actor_cells,
                    },
                );

                info!("New topic subscription created: {}", topic_name);
            }
        }
        drop(subscriptions); // Release the write lock

        // Notify host about new topic subscription if this is the first subscription for this topic
        if is_new_topic {
            self.notify_host_of_subscription(topic_name, true).await;
        }

        // Join the cluster-wide process group for this topic if we have an actor cell
        if let Some(cell) = actor_cell {
            self.join_process_group(topic_name, cell).await;
        }

        Ok(())
    }

    /// Register all existing subscriptions with the host after connection is established
    async fn register_existing_subscriptions_with_host(&self) {
        let subscriptions = self.local_subscriptions.read().await;
        let topics: Vec<String> = subscriptions.keys().cloned().collect();
        drop(subscriptions); // Release the read lock

        if !topics.is_empty() {
            info!(
                "Registering {} existing subscriptions with host: {:?}",
                topics.len(),
                topics
            );

            for topic in topics {
                self.notify_host_of_subscription(&topic, true).await;
                // Small delay to avoid overwhelming the host
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }

    /// Notify host about subscription/unsubscription
    async fn notify_host_of_subscription(&self, topic_name: &str, subscribe: bool) {
        let registration_msg = ClusterMessage::SubscriptionRegistration {
            client_id: self.client_id.clone(),
            topic_name: topic_name.to_string(),
            subscribe,
        };

        // Get host communication forwarders
        let host_forwarders = ractor::pg::get_members(&"cluster_communication".to_string())
            .into_iter()
            .filter(|actor| !actor.get_id().is_local()) // Only remote (host) forwarders
            .collect::<Vec<_>>();

        if !host_forwarders.is_empty() {
            info!(
                "Notifying host about {} for topic '{}' from client '{}'",
                if subscribe {
                    "subscription"
                } else {
                    "unsubscription"
                },
                topic_name,
                self.client_id
            );

            for host_forwarder in host_forwarders {
                let forwarder_ref = ActorRef::<ClusterMessage>::from(host_forwarder.clone());
                if let Err(e) = forwarder_ref.cast(registration_msg.clone()) {
                    error!(
                        "Failed to send subscription registration to host forwarder {:?}: {}",
                        host_forwarder.get_id(),
                        e
                    );
                } else {
                    info!(
                        "✅ Successfully sent subscription registration to host forwarder: {:?}",
                        host_forwarder.get_id()
                    );
                }
            }
        } else {
            warn!("No host forwarders found to notify about subscription - retrying in 1 second");
            // Retry after a short delay
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            self.retry_subscription_notification(topic_name, subscribe)
                .await;
        }
    }

    /// Retry subscription notification with exponential backoff
    async fn retry_subscription_notification(&self, topic_name: &str, subscribe: bool) {
        for attempt in 1..=3 {
            let host_forwarders = ractor::pg::get_members(&"cluster_communication".to_string())
                .into_iter()
                .filter(|actor| !actor.get_id().is_local())
                .collect::<Vec<_>>();

            if !host_forwarders.is_empty() {
                info!(
                    "Retry {} successful: found host forwarders for topic '{}' (attempt {})",
                    if subscribe {
                        "subscription"
                    } else {
                        "unsubscription"
                    },
                    topic_name,
                    attempt
                );
                self.send_subscription_to_forwarders(topic_name, subscribe, &host_forwarders)
                    .await;
                return;
            } else {
                warn!(
                    "Retry {} failed: no host forwarders found (attempt {})",
                    attempt, 3
                );
                if attempt < 3 {
                    tokio::time::sleep(tokio::time::Duration::from_secs(attempt)).await;
                }
            }
        }
        error!("Failed to notify host about subscription after 3 attempts");
    }

    /// Send subscription message to host forwarders
    async fn send_subscription_to_forwarders(
        &self,
        topic_name: &str,
        subscribe: bool,
        host_forwarders: &[ractor::ActorCell],
    ) {
        let registration_msg = ClusterMessage::SubscriptionRegistration {
            client_id: self.client_id.clone(),
            topic_name: topic_name.to_string(),
            subscribe,
        };

        for host_forwarder in host_forwarders {
            let forwarder_ref = ActorRef::<ClusterMessage>::from(host_forwarder.clone());
            if let Err(e) = forwarder_ref.cast(registration_msg.clone()) {
                error!(
                    "Failed to send subscription registration to host forwarder {:?}: {}",
                    host_forwarder.get_id(),
                    e
                );
            } else {
                info!(
                    "✅ Successfully sent subscription registration to host forwarder: {:?}",
                    host_forwarder.get_id()
                );
            }
        }
    }

    /// Join a process group for cluster-wide communication
    async fn join_process_group(&self, topic_name: &str, actor_cell: ractor::ActorCell) {
        let group_name = format!("topic_{}", topic_name);

        info!(
            "Joining process group '{}' for cluster-wide communication",
            group_name
        );

        ractor::pg::join(group_name.clone(), vec![actor_cell]);
        info!("Successfully joined process group '{}'", group_name);
    }

    /// Process internal events in the runtime
    async fn process_internal_event(
        &self,
        event: InternalEvent,
    ) -> Result<(), crate::error::Error> {
        debug!("Received internal event: {event:?}");
        match event {
            InternalEvent::ProtocolEvent(event) => {
                self.process_protocol_event(event).await?;
            }
            InternalEvent::Shutdown => {
                self.shutdown_flag.store(true, Ordering::SeqCst);
                self.shutdown_notify.notify_waiters();
            }
        }
        Ok(())
    }

    /// Forward protocol events to external channel
    async fn process_protocol_event(&self, event: Event) -> Result<(), crate::error::Error> {
        match event {
            Event::PublishMessage {
                topic_type,
                topic_name,
                message,
            } => {
                self.handle_publish_message_local(&topic_name, topic_type, message)
                    .await?;
            }
            _ => {
                // Other protocol events are sent to external
                self.external_tx
                    .send(event)
                    .await
                    .map_err(RuntimeError::EventError)?;
            }
        }
        Ok(())
    }

    /// Start the internal event processing loop
    async fn event_loop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut internal_rx = self
            .internal_rx
            .lock()
            .await
            .take()
            .ok_or("Internal receiver already taken")?;

        info!("ClusterClientRuntime event loop starting");

        loop {
            tokio::select! {
                Some(event) = internal_rx.recv() => {
                    debug!("Processing internal event");

                    if matches!(event, InternalEvent::Shutdown) {
                        info!("Received shutdown event");
                        self.process_internal_event(event).await?;
                        break;
                    }

                    if let Err(e) = self.process_internal_event(event).await {
                        error!("Error processing internal event: {e}");
                        break;
                    }
                }
                _ = self.shutdown_notify.notified() => {
                    if self.shutdown_flag.load(Ordering::SeqCst) {
                        info!("Runtime received shutdown notification");
                        break;
                    }
                }
                else => {
                    warn!("Internal event channel closed");
                    break;
                }
            }
        }

        // Drain remaining events
        info!("Draining remaining events before shutdown");
        while let Ok(event) = internal_rx.try_recv() {
            if let Err(e) = self.process_internal_event(event).await {
                error!("Error processing event during shutdown: {e}");
            }
        }

        info!("ClusterClientRuntime event loop stopped");
        Ok(())
    }

    /// Initialize the node server for cluster communication
    async fn init_node_server(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let node = NodeServer::new(
            self.port,
            self.cookie.clone(),
            format!("{}@{}:{}", self.node_name, self.host, self.port),
            self.host.clone(),
            Some(IncomingEncryptionMode::Raw),
            Some(NodeConnectionMode::Isolated),
        );

        let (node_ref, _node_handle) = Actor::spawn(None, node, ())
            .await
            .map_err(|e| format!("Failed to start node server: {}", e))?;

        *self.node_ref.lock().await = Some(node_ref);
        info!(
            "ClusterClientRuntime node server initialized on {}:{}",
            self.host, self.port
        );

        Ok(())
    }

    /// Initialize cluster communication forwarder for client
    async fn init_cluster_communication(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing cluster communication forwarder for client");

        let forwarder = ClusterClientForwarder {
            runtime: Arc::new(self.clone()),
        };

        let (forwarder_ref, _handle) = ractor::Actor::spawn(None, forwarder, ())
            .await
            .map_err(|e| format!("Failed to start cluster forwarder: {}", e))?;

        ractor::pg::join(
            "cluster_communication".to_string(),
            vec![forwarder_ref.get_cell()],
        );

        info!(
            "✅ ClusterClientRuntime communication forwarder initialized and joined process group"
        );
        Ok(())
    }
}

#[async_trait]
impl Runtime for ClusterHostRuntime {
    fn id(&self) -> RuntimeID {
        self.id
    }

    async fn subscribe_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
    ) -> Result<(), RuntimeError> {
        // Extract actor cell for cluster operations before subscribing
        let actor_cell = actor
            .as_any()
            .downcast_ref::<ActorRef<Task>>()
            .map(|task_actor| task_actor.get_cell());

        self.handle_global_subscribe(topic_name, topic_type, actor, actor_cell, None)
            .await
    }

    async fn publish_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        self.handle_publish_message_global(topic_name, topic_type, message)
            .await
    }

    async fn tx(&self) -> mpsc::Sender<Event> {
        // Create an intercepting sender that routes events through internal processing
        let internal_tx = self.internal_tx.clone();
        let (interceptor_tx, mut interceptor_rx) = mpsc::channel::<Event>(DEFAULT_CHANNEL_BUFFER);

        tokio::spawn(async move {
            while let Some(event) = interceptor_rx.recv().await {
                if let Err(e) = internal_tx.send(InternalEvent::ProtocolEvent(event)).await {
                    error!("Failed to forward event to internal channel: {e}");
                    break;
                }
            }
        });

        interceptor_tx
    }

    async fn transport(&self) -> Arc<dyn Transport> {
        Arc::clone(&self.transport)
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        self.external_rx
            .lock()
            .await
            .take()
            .map(ReceiverStream::new)
    }

    async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Starting ClusterHostRuntime {} on {}:{}",
            self.id, self.host, self.port
        );

        // Initialize the node server
        self.init_node_server().await?;

        // Initialize cluster communication forwarder for this host
        self.init_cluster_communication().await?;

        // Start the event loop
        self.event_loop().await
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initiating cluster host runtime shutdown for {}", self.id);

        // Send shutdown signal
        self.internal_tx
            .send(InternalEvent::Shutdown)
            .await
            .map_err(|e| format!("Failed to send shutdown signal: {e}"))?;

        // Stop the node server if it exists
        if let Some(ref node) = *self.node_ref.lock().await {
            node.stop(Some("Runtime shutdown".to_string()));
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }

    #[cfg(feature = "cluster")]
    async fn connect_to_remote(
        &self,
        _remote_addr: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Host doesn't connect to remotes - clients connect to host
        Err("ClusterHostRuntime doesn't connect to remotes. Use ClusterClientRuntime for client connections.".into())
    }
}

#[async_trait]
impl Runtime for ClusterClientRuntime {
    fn id(&self) -> RuntimeID {
        self.id
    }

    async fn subscribe_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
    ) -> Result<(), RuntimeError> {
        // Extract actor cell for cluster operations before subscribing
        let actor_cell = actor
            .as_any()
            .downcast_ref::<ActorRef<Task>>()
            .map(|task_actor| task_actor.get_cell());

        self.handle_local_subscribe(topic_name, topic_type, actor, actor_cell)
            .await
    }

    async fn publish_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        self.handle_publish_message_local(topic_name, topic_type, message)
            .await
    }

    async fn tx(&self) -> mpsc::Sender<Event> {
        // Create an intercepting sender that routes events through internal processing
        let internal_tx = self.internal_tx.clone();
        let (interceptor_tx, mut interceptor_rx) = mpsc::channel::<Event>(DEFAULT_CHANNEL_BUFFER);

        tokio::spawn(async move {
            while let Some(event) = interceptor_rx.recv().await {
                if let Err(e) = internal_tx.send(InternalEvent::ProtocolEvent(event)).await {
                    error!("Failed to forward event to internal channel: {e}");
                    break;
                }
            }
        });

        interceptor_tx
    }

    async fn transport(&self) -> Arc<dyn Transport> {
        Arc::clone(&self.transport)
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        self.external_rx
            .lock()
            .await
            .take()
            .map(ReceiverStream::new)
    }

    async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Starting ClusterClientRuntime {} on {}:{}",
            self.id, self.host, self.port
        );

        // Initialize the node server
        self.init_node_server().await?;

        // Connect to cluster host
        self.connect_to_host().await?;

        // Start the event loop
        self.event_loop().await
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initiating cluster client runtime shutdown for {}", self.id);

        // Send shutdown signal
        self.internal_tx
            .send(InternalEvent::Shutdown)
            .await
            .map_err(|e| format!("Failed to send shutdown signal: {e}"))?;

        // Stop the node server if it exists
        if let Some(ref node) = *self.node_ref.lock().await {
            node.stop(Some("Runtime shutdown".to_string()));
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }

    #[cfg(feature = "cluster")]
    async fn connect_to_remote(
        &self,
        remote_addr: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // For clients, connecting to remote means connecting to host
        if remote_addr != self.host_address {
            return Err("ClusterClientRuntime can only connect to its configured host".into());
        }
        self.connect_to_host().await
    }
}

/// Host forwarder for distributing messages to all connected clients
#[cfg(feature = "cluster")]
#[derive(Debug, Clone)]
struct ClusterHostForwarder {
    runtime: Arc<ClusterHostRuntime>,
}

#[cfg(feature = "cluster")]
#[async_trait]
impl ractor::Actor for ClusterHostForwarder {
    type Msg = ClusterMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ractor::ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ractor::ActorProcessingErr> {
        info!("[ClusterHostForwarder] Starting up and joining cluster communication groups");
        ractor::pg::join("cluster_communication".to_string(), vec![myself.get_cell()]);
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ractor::ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ractor::ActorProcessingErr> {
        debug!("[ClusterHostForwarder] Processing cluster message from client");

        match message {
            ClusterMessage::Task {
                task,
                source_agent_id,
                target_topic,
            } => {
                info!(
                    "[ClusterHostForwarder] Received task from agent '{:?}' for topic '{}': {}",
                    source_agent_id,
                    target_topic,
                    task.prompt.chars().take(100).collect::<String>()
                );

                // Use global method to distribute to all clients
                let message_arc = Arc::new(task) as Arc<dyn Any + Send + Sync>;
                if let Err(e) = self
                    .runtime
                    .handle_publish_message_global(&target_topic, TypeId::of::<Task>(), message_arc)
                    .await
                {
                    error!(
                        "Failed to distribute cluster message to global topic '{}': {}",
                        target_topic, e
                    );
                } else {
                    info!(
                        "[ClusterHostForwarder] Successfully distributed task to global topic '{}'",
                        target_topic
                    );
                }
            }
            ClusterMessage::SubscriptionRegistration {
                client_id,
                topic_name,
                subscribe,
            } => {
                info!(
                    "[ClusterHostForwarder] Client '{}' {} topic '{}'",
                    client_id,
                    if subscribe {
                        "subscribing to"
                    } else {
                        "unsubscribing from"
                    },
                    topic_name
                );

                self.runtime
                    .handle_subscription_registration(client_id, topic_name, subscribe)
                    .await;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cluster")]
impl ClusterHostForwarder {}

/// Client forwarder for receiving messages from host and routing to local subscribers
#[cfg(feature = "cluster")]
#[derive(Debug, Clone)]
struct ClusterClientForwarder {
    runtime: Arc<ClusterClientRuntime>,
}

#[cfg(feature = "cluster")]
#[async_trait]
impl ractor::Actor for ClusterClientForwarder {
    type Msg = ClusterMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ractor::ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ractor::ActorProcessingErr> {
        info!("[ClusterClientForwarder] Starting up and joining cluster communication groups");
        ractor::pg::join("cluster_communication".to_string(), vec![myself.get_cell()]);
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ractor::ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ractor::ActorProcessingErr> {
        info!("[ClusterClientForwarder] Processing cluster message from host");

        match message {
            ClusterMessage::Task {
                task,
                source_agent_id,
                target_topic,
            } => {
                info!(
                    "[ClusterClientForwarder] Received task from agent '{:?}' for topic '{}': {}",
                    source_agent_id,
                    target_topic,
                    task.prompt.chars().take(100).collect::<String>()
                );

                // Use local-only method to send to local subscribers without forwarding back to host
                let message_arc = Arc::new(task) as Arc<dyn Any + Send + Sync>;
                if let Err(e) = self
                    .runtime
                    .deliver_to_local_subscribers_only(
                        &target_topic,
                        TypeId::of::<Task>(),
                        message_arc,
                    )
                    .await
                {
                    error!(
                        "Failed to forward cluster message to local topic '{}': {}",
                        target_topic, e
                    );
                } else {
                    info!(
                        "[ClusterClientForwarder] Successfully delivered task to local topic '{}'",
                        target_topic
                    );
                }
            }
            ClusterMessage::SubscriptionRegistration { .. } => {
                // Client forwarders should not receive subscription registrations from host
                // These are sent FROM clients TO host, not the other way around
                warn!("[ClusterClientForwarder] Unexpected SubscriptionRegistration message from host");
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cluster")]
impl ClusterClientForwarder {}

/*
Working Example for distributed using ractor_cluster
use clap::{Parser, Subcommand};
use ractor::{Actor, ActorProcessingErr, ActorRef, SupervisionEvent};
use ractor_cluster::node::{
    NodeConnectionMode, NodeServer, NodeServerMessage, NodeServerSessionInformation,
};
use ractor_cluster::{
    IncomingEncryptionMode, NodeEventSubscription, RactorClusterMessage, client_connect,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(RactorClusterMessage)]
pub enum PingPongMessage {
    Ping(u32),
    Pong(u32),
}

#[derive(Debug, Clone)]
pub struct PingPongActor {
    id: String,
    is_producer: bool,
}

impl PingPongActor {
    pub fn new_producer(id: String) -> Self {
        PingPongActor { id, is_producer: true }
    }

    pub fn new_consumer(id: String) -> Self {
        PingPongActor { id, is_producer: false }
    }
}

impl Actor for PingPongActor {
    type Msg = PingPongMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        println!("[{}] Starting up and joining 'pingpong' process group", self.id);

        // Join the process group to make this actor discoverable across the cluster
        ractor::pg::join("pingpong".to_string(), vec![myself.get_cell()]);

        Ok(())
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let group = "pingpong".to_string();

        // Get all remote actors in the process group
        let remote_actors = ractor::pg::get_members(&group)
            .into_iter()
            .filter(|actor| !actor.get_id().is_local())
            .map(ActorRef::<Self::Msg>::from)
            .collect::<Vec<_>>();

        match message {
            PingPongMessage::Ping(n) => {
                if self.is_producer {
                    println!("[Producer {}] Sending Ping({}) to {} remote actors",
                             self.id, n, remote_actors.len());

                    // Send ping to all remote actors
                    for remote_actor in remote_actors {
                        remote_actor.cast(PingPongMessage::Ping(n))?;
                    }
                } else {
                    println!("[Consumer {}] Received Ping({}) from remote producer!", self.id, n);
                    println!("[Consumer {}] Sending Pong({}) back to {} remote actors",
                             self.id, n, remote_actors.len());

                    // Send pong back to all remote actors
                    for remote_actor in remote_actors {
                        remote_actor.cast(PingPongMessage::Pong(n))?;
                    }
                }
            }
            PingPongMessage::Pong(n) => {
                if self.is_producer {
                    println!("[Producer {}] Received Pong({}) from remote consumer!", self.id, n);
                } else {
                    println!("[Consumer {}] Received Pong({}) (unexpected)", self.id, n);
                }
            }
        }

        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: SupervisionEvent,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            SupervisionEvent::ActorTerminated(actor_id, _, _) => {
                println!("[{} {}] Supervised actor {:?} terminated",
                         if self.is_producer { "Producer" } else { "Consumer" },
                         self.id, actor_id);
            }
            _ => {}
        }
        Ok(())
    }

    async fn post_stop(
        &self,
        _myself: ActorRef<Self::Msg>,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        println!("[{} {}] Shutting down",
                 if self.is_producer { "Producer" } else { "Consumer" },
                 self.id);
        Ok(())
    }
}

// Event handler for tracking node connections
#[derive(Debug, Clone)]
pub struct ClusterEventHandler {
    actor_ref: ActorRef<PingPongMessage>,
    is_producer: bool,
}

impl ClusterEventHandler {
    pub fn new(actor_ref: ActorRef<PingPongMessage>, is_producer: bool) -> Self {
        Self { actor_ref, is_producer }
    }
}

impl NodeEventSubscription for ClusterEventHandler {
    fn node_session_opened(&self, session_info: NodeServerSessionInformation) {
        println!(
            "{}: Node session opened to/from {}",
            if self.is_producer { "Producer" } else { "Consumer" },
            session_info.peer_addr
        );
    }

    fn node_session_disconnected(&self, session_info: NodeServerSessionInformation) {
        println!(
            "{}: Node session disconnected from {}",
            if self.is_producer { "Producer" } else { "Consumer" },
            session_info.peer_addr
        );
    }

    fn node_session_authenicated(&self, session_info: NodeServerSessionInformation) {
        println!(
            "{}: Node session authenticated with {}",
            if self.is_producer { "Producer" } else { "Consumer" },
            session_info.peer_addr
        );
        println!(
            "{}: ✅ Distributed cluster ready for message passing!",
            if self.is_producer { "Producer" } else { "Consumer" }
        );

        // If this is a producer, start sending pings after a delay
        if self.is_producer {
            let actor_ref = self.actor_ref.clone();

            tokio::spawn(async move {
                // Wait a bit for the cluster to stabilize
                sleep(Duration::from_secs(2)).await;

                for i in 1..=5 {
                    sleep(Duration::from_secs(2)).await;
                    println!("Producer: Sending Ping({}) to remote consumer(s)", i);
                    let _ = actor_ref.cast(PingPongMessage::Ping(i));
                }
            });
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run as a producer node
    Producer {
        #[arg(short = 'p', long, default_value = "9001")]
        port: u16,
        #[arg(short = 'c', long, default_value = "localhost:9002")]
        consumer_addr: String,
    },
    /// Run as a consumer node
    Consumer {
        #[arg(short = 'p', long, default_value = "9002")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Commands::Producer {
            port,
            consumer_addr,
        } => {
            run_producer_node(port, consumer_addr).await?;
        }
        Commands::Consumer { port } => {
            run_consumer_node(port).await?;
        }
    }

    Ok(())
}

async fn run_producer_node(
    port: u16,
    consumer_addr: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Producer Node on port {}", port);
    println!("🎯 Will connect to consumer at: {}", consumer_addr);
    println!("=====================================");

    // Create node server
    let node = NodeServer::new(
        port,
        "cluster-cookie".to_string(),
        format!("producer@localhost:{}", port),
        "localhost".to_string(),
        Some(IncomingEncryptionMode::Raw),
        Some(NodeConnectionMode::Isolated),
    );

    let (node_ref, _node_handle) = Actor::spawn(None, node, ())
        .await
        .expect("Failed to start node server");

    println!("✅ Producer node started successfully\n");

    // Create and spawn producer actor
    let producer = PingPongActor::new_producer("producer".to_string());
    let (producer_ref, _producer_handle) = Actor::spawn(
        Some("producer".to_string()),
        producer,
        (),
    )
    .await
    .expect("Failed to spawn producer");

    // Subscribe to node events
    let subscription = ClusterEventHandler::new(producer_ref, true);
    node_ref.send_message(NodeServerMessage::SubscribeToEvents {
        id: "producer-events".to_string(),
        subscription: Box::new(subscription),
    })?;

    // Attempt to connect to the consumer node
    tokio::spawn(async move {
        // Wait a moment for the node server to be fully ready
        sleep(Duration::from_secs(1)).await;

        println!("[Producer] Attempting to connect to consumer at {}", consumer_addr);

        match client_connect(&node_ref, consumer_addr.as_str()).await {
            Ok(()) => {
                println!("[Producer] Successfully initiated connection to consumer");
            }
            Err(e) => {
                println!("[Producer] Failed to connect to consumer: {}", e);
            }
        }
    });

    println!("✅ Producer actor spawned successfully");
    println!("📡 Producer ready for distributed cluster communication!\n");

    // Keep running indefinitely
    println!("🔄 Producer running indefinitely. Press Ctrl+C to stop...");
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for event");

    Ok(())
}

async fn run_consumer_node(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Consumer Node on port {}", port);
    println!("👂 Listening for producer connections...");
    println!("=====================================");

    // Create node server
    let node = NodeServer::new(
        port,
        "cluster-cookie".to_string(),
        format!("consumer@localhost:{}", port),
        "localhost".to_string(),
        Some(IncomingEncryptionMode::Raw),
        Some(NodeConnectionMode::Isolated),
    );

    let (node_ref, _node_handle) = Actor::spawn(None, node, ())
        .await
        .expect("Failed to start node server");

    println!("✅ Consumer node started successfully\n");

    // Create and spawn consumer actor
    let consumer = PingPongActor::new_consumer("consumer".to_string());
    let (consumer_ref, _consumer_handle) = Actor::spawn(
        Some("consumer".to_string()),
        consumer,
        (),
    )
    .await
    .expect("Failed to spawn consumer");

    // Subscribe to node events
    let subscription = ClusterEventHandler::new(consumer_ref, false);
    node_ref.send_message(NodeServerMessage::SubscribeToEvents {
        id: "consumer-events".to_string(),
        subscription: Box::new(subscription),
    })?;

    println!("✅ Consumer actor spawned successfully");
    println!("📡 Consumer ready for distributed cluster communication!\n");

    // Keep running indefinitely
    println!("🔄 Consumer running indefinitely. Press Ctrl+C to stop...");
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for event");

    Ok(())
}
*/
