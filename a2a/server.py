"""
A2A (Agent-to-Agent) Protocol — based on Google's open spec
Every agent exposes:
  GET /.well-known/agent.json  → AgentCard (capabilities, input/output schema)
  POST /a2a/invoke             → Execute a task
  GET /a2a/agents              → List all known agents
"""

from ast import ClassDef
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


a2a = FastAPI(title="OmniAgent A2A Server")


#------------protocal methods -------------------------
class AgentCard(BaseModel):
    agent_id: str
    name:str
    version: str
    description:str
    capabilities:list[str]
    input_modes:list[str] = ["text"]
    output_modes:list[str] = ["text", "json"]
    endpoint:str

class A2ATask(BaseModel):
    task_id:        str = ""
    agent_id:       str
    task:           str
    context:        Dict = {}
    callback_url:   Optional[str] = None

class A2AResult(BaseModel):
    task_id: str
    agent_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = {}
    

# ---- Agent Registry ---------------------------------
REGISTRY: dict[str, AgentCard] = {
    "claude_code": AgentCard(
        agent_id="claude_code",
        name="Claude Code Agent",
        description="Code generation, debugging, architecture using Claude Sonnet",
        capabilities=["code_gen","code_review","debugging","architecture"],
        endpoint="http://localhost:8001/a2a/invoke"

    ),
    "google_research":AgentCard(
        agent_id="google_research",
        name="Google ADK Research Agent",
        description="Live web research with Google Search grounding via ADK + Vertex AI",
        capabilities=["web_research","fact_check","report_gen","data_analysis"],
        endpoint="http://localhost:8001/a2a/invoke"
    ),
    "bedrock_data"AgentCard(
        agent_id="bedrock_data",
        name="AWS Bedrock Data Agent",
        description="SQL generation, data analysis, statistical insights via AWS Bedrock",
        capabilities=["sql_gen","data_analysis","charts","forecasting"],
        endpoint="http://localhost:8001/a2a/invoke"
    ),
    "azure_docs": AgentCard(
        agent_id="azure_docs", name="Azure AI Foundry Doc Agent",
        description="Document extraction, summarization, classification via Azure AI Foundry",
        capabilities=["doc_extract","summarize","classify","translate"],
        endpoint="http://localhost:8001/a2a/invoke"
    )

}

@a2a.get("/.well-know/agent.json", response_model=AgentCard)
async def orchestrator_card():
    """
    This orchestrator's AgentCard for A2A discovery
    """
    return AgentCard(
        agent_id="omni_orchestrator",
        name="OmniAgent Orchestrator",
        description="Multi-agent task routing and workflow management",
        capabilities=["task_dispatch","workflow_orchestration","agent_selection"],
        endpoint="http://localhost:8001/a2a/invoke"
    )

@a2a.get("/a2a/agents")
async def list_agents():
    return list(REGISTRY.values())

@a2a.post("/a2a/invoke", response_model=A2AResult)
async def invoke_agent(task:A2ATask):
    task_id = task.task_id or str(uuid.uuid4)
    try:
        if task.agent_id == "claude_code":
            from agents.claude_agent import ClaudeCodeAgent
            result = await ClaudeCodeAgent().run(task.task, task.context)
        elif task.agent_id == "google_research":
            from agents.google_adk_agent import GoogleADKResearchAgent
            result = await GoogleADKResearchAgent().research(task.task)
        elif task.agent_id == "bedrock_data":
            from agents.bedrock_agent import BedrockDataAgent
            result = await BedrockDataAgent().analyze(task.task, task.context)
        elif task.agent_id == "azure_docs":
            from agents.azure_agent import AzureDocAgent
            result = await AzureDocAgent().process(task.task, task.context)
        else:
            raise ValueError(f"Unknown agent: {task.agent_id}")
        return A2AResult(task_id=task_id, agent_id=task.agent_id, status="completed", result=result)
    except Exception as e:
        return A2AResult(task_id=task_id, agent_id=task.agent_id, status="failure", error=str(e))

    