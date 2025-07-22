# gRPC Hello World Example

This example demonstrates the simplest use case of the gRPC runtime - a single agent that responds to messages sent via gRPC.

## Overview

This example consists of:
- **Server**: A gRPC runtime server hosting a simple greeter agent
- **Client**: A command-line client that sends messages to the server

## Prerequisites

Make sure you have your OpenAI API key set:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Example

### Step 1: Start the Server

In one terminal, start the gRPC server:

```bash
cargo run --package grpc-hello --bin grpc-server
```

You should see:
```
Starting gRPC Hello World Example
=================================
Creating gRPC runtime on 127.0.0.1:50051...
✅ Greeter agent registered!

🚀 gRPC server is ready!
Listening for connections...

Topics:
  - greetings: Send greeting requests
  - questions: Ask simple questions

Press Ctrl+C to stop the server
```

### Step 2: Run the Client

In another terminal, run the client:

```bash
cargo run --package grpc-hello --bin grpc-client
```

The client will:
1. Connect to the gRPC server
2. Send a greeting request
3. Ask a simple question
4. Send another greeting request

You'll see the responses in the server terminal.

## How It Works

1. **gRPC Runtime**: The server creates a `GrpcRuntime` that listens on port 50051
2. **Agent Registration**: A simple greeter agent is registered with the runtime
3. **Topic Subscription**: The agent subscribes to "greetings" and "questions" topics
4. **Client Connection**: The client connects via gRPC and publishes messages to these topics
5. **Processing**: The agent processes the messages using the configured LLM
6. **Response**: Results are displayed in the server console

## Architecture

```
┌─────────────┐         gRPC          ┌─────────────────┐
│   Client    │◄─────────────────────►│  gRPC Runtime   │
│             │      (HTTP/2)          │                 │
└─────────────┘                        │  ┌───────────┐  │
                                       │  │  Greeter  │  │
                                       │  │   Agent   │  │
                                       │  └───────────┘  │
                                       └─────────────────┘
```

## Key Concepts

- **gRPC Runtime**: Provides network-accessible runtime for agents
- **Protocol Buffers**: Efficient binary protocol for communication
- **Topics**: Pub/sub mechanism for routing messages to agents
- **Event Stream**: Real-time event notifications from the runtime

## Extending the Example

You can extend this example by:
1. Adding more agents with different capabilities
2. Creating interactive clients that accept user input
3. Implementing bidirectional communication
4. Adding authentication and TLS
5. Deploying to separate machines

## Troubleshooting

- **Connection Refused**: Make sure the server is running before starting the client
- **No Response**: Check that your OPENAI_API_KEY is set correctly
- **Port Already in Use**: Another process might be using port 50051