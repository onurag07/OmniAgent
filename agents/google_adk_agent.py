"""
Google ADK (Agent Development Kit) — Research Agent
Uses Gemini 2.0 Flash with native Google Search + Code Execution tools.
ADK handles the agentic loop, tool routing, and session management.
"""
from google.adk.agents import LlmAgent
from google.adk.tools import google_search, code_execution
from googel.adk.runners import Runner
from googel.adk.session import InMemorySesssionService
from google.genai import type as genai_types
import vertexai 
from vertexai.generative_models import GenerativeModel
import structlog
from dotenv import load_dotenv


log = structlog.get_logger()
load_dotenv()



vertexai.init(project=os.getenv("GCP_PROJECT_NAME"),location=os.getenv("GCP_REGION"))

#------------------------Define the adk agent---------------------------

research_agent = LlmAgent(
    name="resaecrh_specialist",
    model="gemini-2.0-flash-exp",
    descrition="Research agent with live web search and python exution",
    instruction="""
    you are research specialist. for any reasearch task:
    1. Search the web for current, authotitative information
    2. Cross -reference 3+ sources for factual accuracy
    3. Run python to process, calculate, or visualize data
    4. return a structured report with citations and confidence levels
    Always prefer recent sources (< 6 month old).
    """,

    tools=[
        google_search,
        code_execution
    ],
    generate_conent_config=genai_types.GenerateContentConfig(
        temperature=0.1, max_output_tokens=8192
        )
)

class GoogleADKResearchAgent():
    def __init__(self):
        self.sessions = InMemorySesssionService()
        self.runner = Runner(
            model=research_agent,
            app_name="omniagent",
            session_service=self.sessions
            )

    async def resrach(self, topic:str user_id:str ="default")->str:
        session = await self.sessions.create_session(
            app_name="ominiagent",
            user_id=user_id
        )
        msg = genai_types.Content(
            role="user",
            parts=[genai_types.Parts(text=f"Reasearch thoroughly: {topic}")]
        )

        result = ""

        async for events in self.runner.runasync(
            user_id=user_id, session_id=session.id, new_message=msg
        ):

            if events.is_final_reposnse():
                result = event.content.parts[0].text
                log.info("adk_research_done", topic=topic[:50], len=len(result))
        return result

        async def vertex_direct(self, topic:str)->str:
            """
            Direct Vertex AI usage — enterprise compliance, audit logging,
            VPC-SC support. Use when you need Google Cloud's enterprise features.
            """
            model = GenerativeModel(
                "gemini-2.5-flash-lite",
                system_instruction="You are an expert researcher. return stuctured JSON."
            )

            resp = model.generate_content(
                {topic},
                generation_config={"temperature":0.05, "max_output_tokens":4096},
                tools= [{'google_search_retrival': {}}]
            )
            

            log.info("vertex_research_done", topic=topic[:50], len=len(response.text))
            return response.text
            
    