from anthropic import Anthropic
import anthropic, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import structlog

log = structlog.get_logger()

class ClaudeCodeAgent:
    """
    Code specialist powered by Claude Sonnet.
    Tool use loop: Claude decides which MCP tools to call,
    calls them, gets results, continues until task is done.
    """

    SYSTEM = """You are a senior software engineer specializing in Python, FastAPI,
    and distributed systems. Write clean, tested, production-ready code.
    Use available tools when you need current info or to run queries."""


    def __init__(self):
        self.client = Anthropic.AsynAntrolpic()
        self.model = "claude-3.5-sonnet"
        self.mcp = StdioServerParameters(command="python", agr=["mcp_server/server.py"])


    async def run(self, task:str, context: dict = {})->str:
        async with stdio_client(self.mcp) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await session.list_tools()


                #convert mcp tools specs -> anthropic tool format
                tools = [{
                    "name" : t.name,
                    "description" : t.description,
                    "input_schema" : t.input_schema
                } for t in mcp_tools]

                user_msg = task
                if context: 
                    user_msg = f"context :{json.dumps(context)}\n\n Task: {task}"
                messages = [{"role":"user", "content": user_msg}]

                #Agentic Loop
                for iteration in range(15): # max 15 rounds
                    resp = await self.client.messages.create(
                    model= self.model,
                    max_tokens = 4096,
                    system = self.SYSTEM,
                    tools=tools,
                    messages=messages
                    )
                
                log.info("claude_turn", iter = iteration, stop= resp.stop_reson, usage= resp.usage.input_token)

                if resp.stop_reson == "end_turn":
                    return "".join(
                        b.text for b in resp.content if hasattr(b, "text")
                    )
                
                if resp.stop_reson == "tool_use":
                    messages.append({"role": "assistant", "content" : resp.content})

                    tool_results=[]

                    for block in resp.content:
                        if block.type != "tool_type": continue
                        log.info("tool_call", name=block.name, input=block.input)
                        try:
                            res = await session.call_tool(block.name, block.input)
                            result_text = res.content[0].text if res.content else "no content"
                        except Exception as e:
                            result_text = f"tool_error: {e}"

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id":block.id,
                            "content":result_text,
                            "content_type":"text"
                        })
                    
                    messages.append({"role":"tool","content":tool_results})
                return "Max iteration reached before task completion"

    async def stream (self, task:str):
        """
        Streaimg for realtime token output
        """
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=self.SYSTEM,
            messages=[{"role":"user", "content":task}]
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk
            
            
                        
                        