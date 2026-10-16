"""
Azure AI Foundry — Document processing agent.
AI Foundry = Microsoft's unified platform for:
  - Deploying models (GPT-4o, Phi-4, Llama, Mistral)
  - Prompt flows
  - Evaluations
  - Agent deployment
"""
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, UserMessage, AssistantMessage
)
from azure.core.credentials import AzureKeyCredential
import json

class AzureDocAgent:
    """Document extraction + classification + summarization via Azure AI Foundry"""

    def __init__(self):
        # Azure AI Foundry endpoint (Models-as-a-Service)
        self.client = ChatCompletionsClient(
            endpoint="https://YOUR-RESOURCE.services.ai.azure.com/models",
            credential=AzureKeyCredential("AZURE_KEY")
        )
        self.model = "gpt-4o"   # or "Phi-4", "Meta-Llama-3.3-70B-Instruct"

    async def process(self, task: str, context: dict = {}) -> str:
        doc = context.get("document_text", "")
        msgs = [
            SystemMessage(content="You are a document processing expert. Extract structured data precisely."),
            UserMessage(content=f"Document:\n{doc[:8000]}\n\nTask: {task}" if doc else task)
        ]
        resp = self.client.complete(
            messages=msgs,
            model=self.model,
            max_tokens=2048,
            temperature=0.05,
            response_format={"type": "json_object"}  # Force structured JSON
        )
        return resp.choices[0].message.content

    async def extract_structured(self, text: str, schema: dict) -> dict:
        """Force extraction matching a specific JSON schema"""
        prompt = f"""Extract data from this text following EXACTLY this JSON schema:
Schema: {json.dumps(schema, indent=2)}
Text: {text[:4000]}
Return ONLY valid JSON. No explanation."""

        resp = self.client.complete(
            messages=[SystemMessage(content="You are a precise data extractor."),
                      UserMessage(content=prompt)],
            model=self.model,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    async def chat_with_memory(self, messages: list[dict]) -> str:
        """Multi-turn conversation maintaining history"""
        azure_msgs = []
        for m in messages:
            if m["role"] == "system":
                azure_msgs.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                azure_msgs.append(UserMessage(content=m["content"]))
            elif m["role"] == "assistant":
                azure_msgs.append(AssistantMessage(content=m["content"]))
        resp = self.client.complete(messages=azure_msgs, model=self.model, max_tokens=2048)
        return resp.choices[0].message.content