import boto3, json
from botocore.config import Config

class BedrockDataAgent:
    """
    Amazon Bedrock — Data analysis specialist.
    Bedrock advantage: SOC2, HIPAA, FedRAMP compliant models.
    Perfect for enterprise/healthcare/finance use cases.
    """
    def __init__(self):
        self.br  = boto3.client("bedrock-runtime", region_name="us-east-1",
                                 config=Config(retries={"max_attempts":3}))
        self.bra = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        # Using Claude via Bedrock (cross-provider — same model, AWS billing)
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    async def analyze(self, query: str, context: dict = {}) -> str:
        """Use Bedrock Converse API (works across ALL Bedrock models uniformly)"""
        msgs = [{"role":"user","content":[{"text":query}]}]
        if context.get("data"):
            msgs[0]["content"].append({"text":f"\nData: {json.dumps(context['data'])}"})

        resp = self.br.converse(
            modelId=self.model_id,
            messages=msgs,
            system=[{"text":"You are a data analysis expert. Always structure insights clearly."}],
            inferenceConfig={"maxTokens":2048,"temperature":0.1}
        )
        return resp["output"]["message"]["content"][0]["text"]

    def rag_from_kb(self, query: str, kb_id: str) -> str:
        """
        Bedrock AgentCore — Knowledge Base RAG.
        Upload your docs to Bedrock KB → managed embedding + retrieval.
        No separate vector DB needed.
        """
        resp = self.bra.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": kb_id,
                    "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{self.model_id}",
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {"numberOfResults": 10}
                    }
                }
            }
        )
        return resp["output"]["text"]

    def invoke_bedrock_agent(self, agent_id: str, alias_id: str, task: str) -> str:
        """
        Bedrock AgentCore — Managed Agent Runtime.
        Bedrock handles memory, orchestration, tool execution.
        You just define tools in the AWS console.
        """
        stream = self.bra.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId="omniagent-session",
            inputText=task
        )
        result = ""
        for event in stream["completion"]:
            if "chunk" in event:
                result += event["chunk"]["bytes"].decode()
        return result