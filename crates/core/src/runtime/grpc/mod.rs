//! gRPC-based runtime implementation for distributed agent systems
//!
//! This module provides a gRPC runtime that enables agents to communicate
//! across process and network boundaries using Protocol Buffers and gRPC.

pub mod grpc_client;
pub mod grpc_runtime;
pub mod grpc_server;

pub use grpc_client::{GrpcClientConfig, GrpcRuntimeClient};
pub use grpc_runtime::{GrpcRuntime, GrpcRuntimeConfig};
pub use grpc_server::GrpcRuntimeServer;

// Re-export generated protobuf types
pub mod proto {
    tonic::include_proto!("agent_runtime");
}
