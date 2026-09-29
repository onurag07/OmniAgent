# OmniAgent — Multi-Cloud Multi-Agent Orchestration Platform

OmniAgent is an enterprise-grade orchestration platform designed to deploy, manage, and facilitate communication between specialized AI agents across heterogeneous cloud environments. By leveraging LangGraph as the core state machine, it seamlessly routes workloads across Google ADK, Claude SDK, Amazon Bedrock AgentCore, Vertex AI, and Azure AI Foundry.

The platform normalizes tool access via the Model Context Protocol (MCP) and enables asynchronous inter-agent negotiation using Google's Agent-to-Agent (A2A) protocol. A dedicated Context Engine actively manages token budgets, semantic routing, and context window sliding across the multi-agent graph.

## 🏗 Core Architecture

OmniAgent is built on a decentralized multi-agent architecture with a centralized context and orchestration layer:

- **Orchestration Layer (LangGraph):** Manages the global state and defines the routing logic (DAG) between cloud-specific agents. Handles state persistence, error recovery, and graph checkpoints.
- **Context Engine:** A standalone service that monitors token consumption per agent. It implements dynamic context compression (summarization) and sliding window techniques to prevent token exhaustion during extensive A2A communication cycles.
- **Unified Tooling (MCP):** Agents share standard tools (RAG retrievers, API clients, DB executors) exposed via decoupled MCP servers, ensuring consistent tool schemas across Google, Anthropic, Bedrock, and Azure models.
- **Inter-Agent Communication (A2A):** Agents broadcast capabilities and delegate sub-tasks using the A2A protocol, allowing a Vertex-based reasoning agent to offload creative drafting to a Bedrock-based Claude agent seamlessly.

## 🛠 Tech Stack

- **Core Framework:** Python 3.11+, LangGraph
- **Cloud SDKs:** Google ADK, Anthropic Python SDK, Amazon Bedrock (boto3/AgentCore), Vertex AI SDK, Azure AI Foundry
- **Protocols:** Model Context Protocol (MCP), Google A2A Protocol
- **State & Memory:** PostgreSQL / Redis (LangGraph Checkpoint persistence)
