fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Always compile proto files when building the crate
    // The grpc feature flag will control whether the module is included at compile time
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/agent_runtime.proto"], &["proto/"])?;
    Ok(())
}
