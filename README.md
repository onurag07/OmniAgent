# OmniAgent — Multi-Cloud Multi-Agent Orchestration Platform

OmniAgent is an enterprise-grade AI orchestration platform that deploys and connects specialized AI agents across heterogeneous cloud environments. It routes tasks to the right agent — Claude on Anthropic, Gemini on Google ADK / Vertex AI, Titan/Nova on Amazon Bedrock, or a model on Azure AI Inference — while maintaining a shared tool layer via the **Model Context Protocol (MCP)** and enabling asynchronous inter-agent delegation through **Google's Agent-to-Agent (A2A) protocol**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Cloud Agents** | Claude (Anthropic), Gemini (Google ADK + Vertex AI), Bedrock, Azure AI Inference |
| **MCP Tooling** | Shared tool server (web search, code execution, DB queries) consumed by all agents |
| **A2A Protocol** | Agents discover each other's capabilities and delegate sub-tasks asynchronously |
| **Context Engine** | Token-budget tracking, dynamic summarization, and sliding-window compression |
| **Agent Harness** | Behavior-driven evaluation suite with automatic LangSmith tracing |
| **LangGraph Orchestration** | DAG-based state machine for routing, checkpointing, and error recovery |

---

## 🗂 Project Structure

```
OmniAgent/
├── agents/
│   ├── claude_agent.py       # Claude Sonnet — code specialist (MCP tool-use loop)
│   ├── google_adk_agent.py   # Gemini 2.0 Flash — research specialist (ADK + Vertex AI)
│   ├── bedrock_agent.py      # Amazon Bedrock — data analysis agent
│   └── azure_agent.py        # Azure AI Inference — document processing agent
├── core/
│   └── context_engine.py     # Token-budget manager, context compression & sliding window
├── mcp_server/
│   └── server.py             # Shared MCP tool server (web search, code exec, DB)
├── a2a/
│   └── server.py             # A2A capability registry & inter-agent delegation server
├── harness/
│   └── agent_harness.py      # Behavior-driven eval harness with LangSmith tracing
├── requirement.txt           # Pinned dependencies
└── .env                      # API keys & cloud config (not committed)
```

---

## 🏗 Architecture Overview

```
                        ┌─────────────────────────┐
                        │   LangGraph Orchestrator  │
                        │  (DAG routing + state)    │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                       │
     ┌────────▼────────┐   ┌────────▼────────┐   ┌─────────▼────────┐
     │  ClaudeCodeAgent │   │GoogleADKResearch│   │  BedrockAgent    │
     │  (Anthropic SDK) │   │  (ADK + Vertex) │   │  (boto3/Bedrock) │
     └────────┬────────┘   └────────┬────────┘   └─────────┬────────┘
              │                      │                       │
              └──────────────────────┼───────────────────────┘
                                     │  MCP (shared tools)
                            ┌────────▼────────┐
                            │   MCP Server     │
                            │ (search/code/db) │
                            └─────────────────┘
                                     │  A2A (inter-agent)
                            ┌────────▼────────┐
                            │   A2A Server     │
                            │ (capability reg) │
                            └─────────────────┘
                                     │
                            ┌────────▼────────┐
                            │  Context Engine  │
                            │ (token budgets)  │
                            └─────────────────┘
```

### Component Details

#### `agents/claude_agent.py` — Code Specialist
- Powered by **Claude 3.5 Sonnet** via the Anthropic Python SDK.
- Implements a **15-round agentic tool-use loop**: Claude decides which MCP tools to call, executes them via `ClientSession`, feeds results back, and repeats until `end_turn`.
- Exposes a `stream()` method for real-time token streaming.

#### `agents/google_adk_agent.py` — Research Specialist
- Built on **Google ADK** (`LlmAgent`) with `google_search` and `code_execution` tools natively wired in.
- Uses **Gemini 2.0 Flash** via ADK's `Runner` + `InMemorySessionService`.
- Also exposes a `vertex_direct()` path that calls **Vertex AI** (`gemini-2.5-flash-lite`) directly for enterprise scenarios requiring VPC-SC, audit logging, or Cloud IAM.

#### `agents/bedrock_agent.py` — Data Analyst
- Calls **Amazon Bedrock** models (Nova/Titan) via `boto3`.

#### `agents/azure_agent.py` — Document Processor
- Uses **Azure AI Inference SDK** for document understanding and entity extraction tasks.

#### `core/context_engine.py` — Context Manager
- Tracks per-agent **token consumption** in real time.
- Applies **dynamic summarization** when an agent approaches its context limit.
- Implements **sliding window** truncation for long multi-turn A2A conversations.

#### `mcp_server/server.py` — Shared Tool Server
- Exposes a unified **MCP-compliant tool schema** (web search, code execution, database queries).
- All agents connect via `stdio_client` — tool definitions are identical regardless of the underlying model provider.

#### `a2a/server.py` — Inter-Agent Delegation
- Implements **Google's A2A protocol** for capability broadcasting and sub-task delegation.
- Allows, e.g., the Vertex research agent to hand off a coding subtask to the Claude agent at runtime.

#### `harness/agent_harness.py` — Evaluation Harness
- **Behavior assertions**: `not_empty`, `returns_json`, `cites_sources`, `contains_code`, `within_latency`, `no_refusal`.
- Automatically traces every run to **LangSmith** via `@traceable`.
- Ships with a default suite covering code generation, research, data analysis, and document processing tasks.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph 1.2, LangGraph Checkpoint (PostgreSQL / Redis) |
| **Agent Frameworks** | Google ADK, Anthropic Python SDK, boto3 / Bedrock AgentCore, Azure AI Inference |
| **LLMs** | Claude 3.5 Sonnet, Gemini 2.0 Flash, Gemini 2.5 Flash Lite (Vertex AI), Bedrock models |
| **Protocols** | MCP 2.2, Google A2A |
| **Tooling** | Tavily (web search), MCP stdio transport |
| **Tracing & Eval** | LangSmith, structlog |
| **Runtime** | Python 3.11+, asyncio, uvicorn, Starlette |

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/<your-org>/OmniAgent.git
cd OmniAgent
python -m venv venv && source venv/bin/activate
pip install -r requirement.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```env
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Cloud / ADK
GCP_PROJECT_NAME=my-project
GCP_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json

# Amazon Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Azure AI
AZURE_AI_ENDPOINT=https://...
AZURE_AI_KEY=...

# LangSmith (optional but recommended for tracing)
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true

# Tavily (web search)
TAVILY_API_KEY=tvly-...
```

### 3. Start the MCP Tool Server

```bash
python mcp_server/server.py
```

### 4. Start the A2A Server

```bash
python a2a/server.py
```

### 5. Run the Evaluation Harness

```bash
python -c "
import asyncio
from harness.agent_harness import AgentHarness
from agents.claude_agent import ClaudeCodeAgent
from agents.google_adk_agent import GoogleADKResearchAgent

harness = AgentHarness()
claude  = ClaudeCodeAgent()
google  = GoogleADKResearchAgent()

agents = {
    'claude_code':     claude.run,
    'google_research': google.research,
}
asyncio.run(harness.run_suite(agents))
"
```

---

## 🧪 Harness Behavior Assertions

| Behavior | Pass Condition |
|---|---|
| `not_empty` | Output > 30 characters |
| `returns_json` | Output is valid JSON |
| `cites_sources` | Contains "source", "according", or "based on" |
| `contains_code` | Contains ` ``` ` or `def ` |
| `within_latency` | Completes within `max_ms` (default 30 s) |
| `no_refusal` | Does not contain "I cannot" / "I'm unable" |

---

## 🔌 Adding a New Agent

1. Create `agents/my_agent.py` implementing an `async def run(task: str) -> str` method.
2. Connect to the MCP server via `stdio_client` for shared tooling.
3. Register capabilities in `a2a/server.py`.
4. Add a `HarnessCase` to `harness/agent_harness.py`.

---

## 📄 License

MIT
