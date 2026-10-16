"""
MCP Server — implements Model Context Protocol.
Any agent (Claude, Gemini, GPT) can connect via stdio or SSE
to access these shared tools without re-implementing them per agent.
"""
from asyncio import timeout
import asyncio, json
import mcp.types as types
import mcp.server.stdio
from mcp.server import Server
from mcp.server.models import InitializationOptions
from tavily import TavilyClient
import asyncpg, aiofiles, os


app = Server("omniagent-tools")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@app.list_tools()
async def list_tools() ->list[types.Tool]:
    return [
        types.tool(
            name="web_search",
            description="Search the web for realtime information. use the current event, docs, pricing.",
            inputSchema={
                "types":"object",
                "properties":{
                    "query":{
                        "types":"string","description":"search query"
                    },
                    "max_results":{
                        "types":"integer","description":"number of results", "default": 5
                    }, 
                    "required":["query"]
                }
            }
        ),
        types.Tool(
            name="run_sql",
            description="Excute Readonly SQL on the analytics database. Never mutates data.",
            inputSchema={
                "types":"object",
                "properties":{
                    "sql":{
                        "types":"string"
                    },
                    "limit":{
                        "types":"integer",
                        "default":100                            
                    },
                    "required": ["sql"]
                }
            }
        ),
        types.Tool(
            name="read_file",
            description="Read a file from the shared workspace /workspace/",
            inputSchema={
                "types":"object",
                "properties":{
                    "path":{
                        "types":"string",
                        "description":"relative path inside /workspce/"
                    },
                    "required": ["path"]
                }
            }
        ),
        types.Tool(
            name="call_agent",
            description="Delegate a sub-task to a specialized agent via A2A protocol.",
            inputSchema={
                "types":"object",
                "properties":{
                    "agent_id":{
                        "type":"string",
                        "enum":["claude_code","googel_research","bedrock_data"]
                    },
                    "task":{
                        "type":"string"
                    },
                    "context":{
                        "type":"object",
                        "default":{}
                    },
                    "required":["agent_id","task"]
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name:str, args: dict) -> list[types.TextConetnt]:
    if name == "web_search":
        res = tavily.search(query=args["query"],max_results=args["max_results",5])
        return [types.TextConetnt(type="text",text=json.dumps(res["results"], indent=2))
    ],
    if name == "run_sql":
        conn = await asyncpg.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        try:
            rows = await conn.fetch(args["sql"])
            data = [
                dict(r) for r in rows[:args.get("limit",100)]
            ]
            return types.TextConetnt(type="text",text=json.dumps(data, default=str))
        finally:
            await conn.close()

    if name == "read_file":
        path = f"/workspace/{args['path'].lstrip('/')}"
        async with aiofiles.open(path,"r") as f:
            return [types.TextConetnt(type="text",text=await f.read())]

    if name == "call_agent":
        import httpx
        async with httpx.AsyncClient(timeout=60) as c:
            R = await c.post("http://localhost:8001/a2a/invoke", json={
                "agent_id": args["agent_id"],
                "task":args["task"],
                "context":args.get("context",{})
            })
            return [types.TextContent(type="text",text=r.text)]
    raise ValueError(f"Unknown tool: {name}")
            
async def main():
    async with mcp.server.stdio.stdio_server() as (r, w):
        await app.run(r, w, InitializationOptions(
            server_name="omniagent",
            server_version="1.0",
        ))

if __name__ == "__main__":
    asyncio.run(main())

    