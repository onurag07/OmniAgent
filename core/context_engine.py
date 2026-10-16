"""
Context Engineering — build optimal context for each agent at each step.
Goal: Maximum signal, minimum noise, always within token budget.
Techniques: relevance ranking, rolling window, compression, routing.
"""
from dataclasses import dataclass, field
from typing import Optional
import tiktoken, json

@dataclass
class ContextBlock:
    content:      str
    source:       str         # "task"|"rag"|"tool_result"|"history"|"agent_memory"
    relevance:    float = 1.0 # 0.0–1.0
    is_required:  bool  = False
    tokens:       int   = 0

@dataclass
class AgentContext:
    agent_id:           str
    system_prompt:      str
    blocks:             list[ContextBlock] = field(default_factory=list)
    model_ctx_window:   int = 128000
    reserved_for_output:int = 4096

    @property
    def budget(self) -> int:
        return self.model_ctx_window - self.reserved_for_output

SYSTEM_PROMPTS = {
    "claude_code":       "You are a senior software engineer. Write clean, tested, production Python.",
    "google_research":   "You are a research analyst. Return fact-based, cited reports.",
    "bedrock_data":      "You are a data scientist. Provide statistical analysis and visualizations.",
    "azure_docs":        "You are a document processor. Extract structured data precisely.",
    "orchestrator":      "You are an AI orchestrator. Delegate tasks efficiently to specialized agents.",
}

class ContextEngine:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def build(
        self,
        agent_id:    str,
        task:        str,
        rag_docs:    list[str]  = [],
        tool_results:list[dict] = [],
        history:     list[dict] = [],
        ctx_window:  int = 128000
    ) -> AgentContext:
        ctx = AgentContext(
            agent_id=agent_id,
            system_prompt=SYSTEM_PROMPTS.get(agent_id, "You are a helpful AI assistant."),
            model_ctx_window=ctx_window
        )

        # 1. Required: task (always included)
        ctx.blocks.append(ContextBlock(
            content=f"# Task\n{task}", source="task",
            is_required=True, tokens=self.tokens(task)
        ))

        # 2. RAG docs (pre-ranked by retriever, add highest relevance first)
        for i, doc in enumerate(rag_docs[:8]):
            ctx.blocks.append(ContextBlock(
                content=doc, source="rag",
                relevance=1.0 - i*0.08,    # decay by rank
                tokens=self.tokens(doc)
            ))

        # 3. Tool results (most recent = most relevant)
        for res in reversed(tool_results[-10:]):
            text = json.dumps(res)
            ctx.blocks.append(ContextBlock(
                content=text, source="tool_result",
                relevance=0.85, tokens=self.tokens(text)
            ))

        # 4. History (rolling window — recency decays relevance)
        for i, msg in enumerate(reversed(history[-20:])):
            text = f"{msg['role']}: {msg['content']}"
            ctx.blocks.append(ContextBlock(
                content=text, source="history",
                relevance=max(0.1, 0.75 - i*0.04),
                tokens=self.tokens(msg["content"])
            ))

        return self._prune(ctx)

    def _prune(self, ctx: AgentContext) -> AgentContext:
        """Fit within token budget: required blocks first, then by relevance"""
        sys_tokens = self.tokens(ctx.system_prompt)
        budget = ctx.budget - sys_tokens

        required = [b for b in ctx.blocks if b.is_required]
        req_tok  = sum(b.tokens for b in required)
        remaining = budget - req_tok

        optional = sorted(
            [b for b in ctx.blocks if not b.is_required],
            key=lambda b: b.relevance, reverse=True
        )
        included = list(required)
        for block in optional:
            if block.tokens <= remaining:
                included.append(block); remaining -= block.tokens
            elif remaining > 150:
                # Truncate to fit
                trunc_toks = self.enc.encode(block.content)[:remaining-20]
                block.content = self.enc.decode(trunc_toks) + " [TRUNCATED]"
                block.tokens  = remaining - 20
                included.append(block); remaining = 0; break
        ctx.blocks = included
        return ctx

    def to_messages(self, ctx: AgentContext) -> list[dict]:
        """Convert to messages format for any LLM"""
        parts = [b.content for b in ctx.blocks]
        return [
            {"role":"system", "content": ctx.system_prompt},
            {"role":"user",   "content": "\n\n---\n\n".join(parts)}
        ]