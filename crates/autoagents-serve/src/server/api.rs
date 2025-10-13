use super::state::AppState;
use crate::workflow::{WorkflowOutput, WorkflowStreamEvent};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteRequest {
    pub input: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteResponse {
    pub success: bool,
    pub output: Option<WorkflowOutput>,
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_time_ms: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListWorkflowsResponse {
    pub workflows: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    log::info!("Creating API router with endpoints:");
    log::info!("  GET  /health");
    log::info!("  GET  /api/v1/workflows");
    log::info!("  POST /api/v1/workflows/:name/execute");

    Router::new()
        .route("/health", get(health))
        .route("/api/v1/workflows", get(list_workflows))
        .route("/api/v1/workflows/{name}/execute", post(execute_workflow))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    log::debug!("Health check endpoint called");
    Json(HealthResponse {
        status: "healthy".to_string(),
    })
}

async fn list_workflows(State(state): State<Arc<AppState>>) -> Json<ListWorkflowsResponse> {
    log::info!("Listing all workflows");
    let workflows = state.list_workflows().await;
    log::debug!("Found {} workflows: {:?}", workflows.len(), workflows);
    Json(ListWorkflowsResponse { workflows })
}

async fn execute_workflow(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(payload): Json<ExecuteRequest>,
) -> Result<Response, AppError> {
    let start = Instant::now();
    log::info!(
        "Executing workflow '{}' with input: '{}'",
        name,
        payload.input
    );

    let workflow = state.get_workflow(&name).await.ok_or_else(|| {
        log::error!("Workflow '{}' not found", name);
        AppError::NotFound(format!("Workflow '{}' not found", name))
    })?;

    log::debug!("Workflow '{}' found, starting execution", name);

    if workflow.stream_enabled() {
        log::debug!("Workflow '{}' streaming enabled", name);

        let stream = workflow
            .run_stream(payload.input.clone())
            .await
            .map_err(|e| {
                log::error!("Workflow '{}' streaming initialization failed: {}", name, e);
                AppError::Internal(e.to_string())
            })?;

        let duration = start.elapsed();
        log::info!(
            "Workflow '{}' streaming initiated in {:.2}s",
            name,
            duration.as_secs_f64()
        );

        let stream_name = name.clone();
        let sse_stream = stream.map(move |item| match item {
            Ok(event) => Ok::<SseEvent, Infallible>(sse_event_from_workflow_event(event)),
            Err(err) => {
                log::error!("Workflow '{}' streaming error: {}", stream_name, err);
                Ok::<SseEvent, Infallible>(sse_error_event(err.to_string()))
            }
        });

        return Ok(Sse::new(sse_stream).into_response());
    }

    match workflow.run(payload.input.clone()).await {
        Ok(output) => {
            let duration = start.elapsed();
            log::info!(
                "Workflow '{}' completed successfully in {:.2}s",
                name,
                duration.as_secs_f64()
            );
            log::debug!("Workflow '{}' output: {:?}", name, output);

            let response = Json(ExecuteResponse {
                success: true,
                output: Some(output),
                error: None,
                execution_time_ms: Some(duration.as_millis() as u64),
            });

            Ok(response.into_response())
        }
        Err(e) => {
            let duration = start.elapsed();
            log::error!(
                "Workflow '{}' failed after {:.2}s: {}",
                name,
                duration.as_secs_f64(),
                e
            );

            let response = Json(ExecuteResponse {
                success: false,
                output: None,
                error: Some(e.to_string()),
                execution_time_ms: Some(duration.as_millis() as u64),
            });

            Ok(response.into_response())
        }
    }
}

fn sse_event_from_workflow_event(event: WorkflowStreamEvent) -> SseEvent {
    let event_name = match event {
        WorkflowStreamEvent::Chunk { .. } => "chunk",
        WorkflowStreamEvent::ToolCall { .. } => "tool_call",
        WorkflowStreamEvent::Complete => "complete",
        WorkflowStreamEvent::ToolCallComplete { .. } => "tool_call_complete",
    };

    let payload = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    SseEvent::default().event(event_name).data(payload)
}

fn sse_error_event(message: String) -> SseEvent {
    SseEvent::default().event("error").data(message)
}

// Error handling
#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": message
        }));

        (status, body).into_response()
    }
}
