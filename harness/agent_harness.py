import asyncio, time, json
from dataclasses import dataclass, field
from langsmith import Client as LS
from langsmith.run_helpers import traceable

@dataclass
class HarnessCase:
    id:          str
    agent_id:    str
    task:        str
    behaviors:   list[str]  # behaviors to assert
    max_ms:      float = 30000
    # behaviors: not_empty|returns_json|cites_sources|within_latency|contains_code

@dataclass
class HarnessResult:
    case_id:   str
    agent_id:  str
    passed:    bool
    behaviors: dict[str, bool]
    latency_ms:float
    output:    str = ""
    error:     str = ""

class AgentHarness:
    def __init__(self):
        self.ls = LS()

    @traceable(name="agent_harness")  # auto-traces to LangSmith
    async def run_case(self, case: HarnessCase, invoke_fn) -> HarnessResult:
        t0 = time.time(); err = ""; out = ""
        try:
            out = await invoke_fn(case.task)
        except Exception as e:
            err = str(e)
        ms = (time.time() - t0) * 1000

        checks = {
            "not_empty":       lambda: len(out.strip()) > 30,
            "returns_json":    lambda: self._is_json(out),
            "cites_sources":   lambda: any(w in out.lower() for w in ["source","according","based on"]),
            "contains_code":   lambda: "```" in out or "def " in out,
            "within_latency":  lambda: ms < case.max_ms,
            "no_refusal":      lambda: not any(w in out.lower() for w in ["i cannot","i'm unable"]),
        }
        beh = {b: checks.get(b, lambda: True)() for b in case.behaviors}
        passed = all(beh.values()) and not err

        self.ls.create_run(
            name=f"harness_{case.id}", run_type="chain",
            inputs={"task":case.task,"agent":case.agent_id},
            outputs={"output":out[:300],"passed":passed,"behaviors":beh},
            extra={"latency_ms":ms,"error":err}
        )
        return HarnessResult(case.id, case.agent_id, passed, beh, ms, out, err)

    def _is_json(self, text: str) -> bool:
        try: json.loads(text); return True
        except: return False

    async def run_suite(self, agents: dict) -> dict:
        cases = [
            HarnessCase("code_001",  "claude_code",     "Write Python binary search with tests", ["not_empty","contains_code"]),
            HarnessCase("res_001",   "google_research",  "Latest developments in AI safety 2025",  ["not_empty","cites_sources"]),
            HarnessCase("data_001",  "bedrock_data",    "Analyze monthly revenue trends",          ["not_empty","within_latency"]),
            HarnessCase("doc_001",   "azure_docs",      "Extract entities from: Alice works at Google since 2020", ["not_empty"]),
        ]
        results = await asyncio.gather(*[
            self.run_case(c, agents[c.agent_id]) for c in cases if c.agent_id in agents
        ])
        passed = sum(1 for r in results if r.passed)
        print(f"\n{'='*50}\nHarness: {passed}/{len(results)} passed\n{'='*50}")
        for r in results:
            status = "✅" if r.passed else "❌"
            print(f"{status} {r.case_id} [{r.agent_id}] {r.latency_ms:.0f}ms")
            for b, ok in r.behaviors.items():
                print(f"   {'✓' if ok else '✗'} {b}")
        return {"total":len(results),"passed":passed,"results":[r.__dict__ for r in results]}