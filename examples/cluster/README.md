# Cluster Example - Three-Tier Host-Client Architecture

This example demonstrates the new three-tier cluster architecture:
- **ClusterHostRuntime**: Dedicated coordinator that manages all client connections and routes events
- **ResearchAgent**: ClusterClientRuntime that gathers information  
- **AnalysisAgent**: ClusterClientRuntime that analyzes research data

Both agents connect to the same host for coordinated cluster communication.

## Running the Example

**Step 1:** Start the dedicated cluster host (this coordinates all communication):

```shell
cargo run -p cluster-example host --port 9000
```

**Step 2:** Start the AnalysisAgent client:

```shell
cargo run -p cluster-example analysis --port 9002 --host-addr localhost:9000
```

**Step 3:** Start the ResearchAgent client:

```shell
cargo run -p cluster-example research --port 9001 --host-addr localhost:9000
```

## Architecture

- **ClusterHostRuntime**: Dedicated host that manages global subscriptions and routes events between all connected clients
- **ClusterClientRuntime (ResearchAgent)**: Connects to host, gathers information, and forwards tasks to AnalysisAgent via the host
- **ClusterClientRuntime (AnalysisAgent)**: Connects to host, receives research data from ResearchAgent, and performs analysis

The host coordinates all inter-client communication, ensuring proper message routing and subscription management across the cluster.
