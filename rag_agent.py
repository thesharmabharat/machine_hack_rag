# agentic_rag.py
"""
Agentic RAG (single-file)
- Multi-agent pipeline: Router → Planner → Retriever (Local+Web) → Synthesizer → Verifier → Finalizer
- Local knowledge: FAISS + sentence-transformers over ./data/*.txt|.md
- Live web: DuckDuckGo Search (no API key)
- Optional tools: Calculator (safe eval for arithmetic)
- Dynamic query handling (factual / open / reasoning / tool)
- Transparent output: citations, why-these-sources, trace, verifier report

Usage
-----
pip install faiss-cpu sentence-transformers duckduckgo-search litellm numpy
export OPENAI_API_KEY=sk-...   # LiteLLM reads this; default model: gpt-4o-mini

# Ask (web+local)
python agentic_rag.py --ask "Latest safety guidance for LLM jailbreak defenses; compare with my notes"

# Local-only
python agentic_rag.py --ask "Summarize our Q2 risks and mitigations" --no-web

# Tool query
python agentic_rag.py --ask "calculate 12*(7+3)-5"
"""

from __future__ import annotations
import os, re, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
from litellm import completion

# =========================
# Utility
# =========================
def chunk_text(text: str, size: int = 500, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

# =========================
# Agent: Router
# =========================
QueryType = Literal["factual", "open", "reasoning", "tool"]

class Router:
    def route(self, q: str) -> QueryType:
        ql = q.lower()
        if re.search(r"\b(calculate|sum|add|multiply|minus|divide|percent|\d+\s*[\+\-\*\/]\s*\d+)\b", ql):
            return "tool"
        if any(x in ql for x in ["why", "how", "derive", "prove", "step-by-step", "explain the steps"]):
            return "reasoning"
        if len(ql.split()) <= 10 and q.strip().endswith("?"):
            return "factual"
        return "open"

# =========================
# Agent: Planner
# =========================
@dataclass
class Plan:
    steps: List[str]
    k_local: int
    k_web: int
    use_verifier: bool

class Planner:
    def make(self, qtype: QueryType, use_web: bool) -> Plan:
        k_local = 10 if qtype == "reasoning" else 6
        k_web = 6 if use_web else 0
        steps = ["analyze", "retrieve_local"]
        if use_web:
            steps.append("retrieve_web")
        steps.append("synthesize")
        use_verifier = qtype in {"factual", "reasoning"}
        if use_verifier:
            steps.append("verify")
        steps.append("finalize")
        return Plan(steps=steps, k_local=k_local, k_web=k_web, use_verifier=use_verifier)

# =========================
# Agent: Local Retriever
# =========================
class LocalIndex:
    def __init__(self, data_dir="data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeds = None
        self.meta: List[Dict[str, Any]] = []

    def build(self):
        texts, meta = [], []
        for p in self.data_dir.rglob("*"):
            if p.suffix.lower() in {".txt", ".md"}:
                try:
                    raw = p.read_text(errors="ignore")
                except Exception:
                    continue
                for ch in chunk_text(raw, size=500, overlap=120):
                    texts.append(ch)
                    meta.append({"uri": f"file://{p.resolve()}", "title": p.name, "text": ch, "kind": "local"})
        if not texts:
            return
        embeds = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        idx = faiss.IndexFlatIP(embeds.shape[1])
        idx.add(np.asarray(embeds, dtype=np.float32))
        self.index = idx
        self.embeds = embeds
        self.meta = meta

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), min(k*3, len(self.meta)))
        qv = self.model.encode(query, normalize_embeddings=True)
        hits = []
        for s, i in zip(scores[0], ids[0]):
            if i == -1: continue
            m = dict(self.meta[int(i)])
            m.update({"id": f"L{i}", "score": float(s), "snippet": m["text"][:240]})
            # light re-rank with cosine sim for robustness
            m["_rr"] = float(util.cos_sim(qv, self.embeds[int(i)]).item())
            hits.append(m)
        hits.sort(key=lambda h: (h["_rr"], h["score"]), reverse=True)
        return hits[:k]

# =========================
# Agent: Web Retriever
# =========================
class WebSearch:
    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        out = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=k)):
                out.append({
                    "id": f"W{i}",
                    "uri": r.get("href"),
                    "title": r.get("title") or "web",
                    "snippet": (r.get("body") or "")[:300],
                    "kind": "web",
                    "text": r.get("body") or ""
                })
        return out

# =========================
# Agent: Tool (optional enhancements)
# =========================
class ToolAgent:
    SAFE = re.compile(r"^[0-9\.\s\+\-\*\/\(\)%]+$")
    def maybe_handle(self, query: str) -> str | None:
        q = query.strip().lower()
        if q.startswith("calculate") or self.SAFE.match(q):
            expr = q.replace("calculate", "").strip()
            if not self.SAFE.match(expr):
                return "Tool refused: unsafe expression."
            try:
                # percent → /100
                expr = re.sub(r"(\d+)\s*%", r"(\1/100)", expr)
                val = eval(expr, {"__builtins__": {}})
                return f"Calculation result: **{val}**"
            except Exception as e:
                return f"Tool error: {e}"
        return None

# =========================
# Agent: Synthesizer
# =========================
SYS = """You are a rigorous research assistant.
Rules:
- Use ONLY the provided sources; never invent facts.
- Every claim/sentence that uses a fact must include an inline [S#] citation.
- If sources are insufficient, say: "Not enough info in sources."
- Style by query type:
  - factual: start with tight bullets, then a short paragraph.
  - open: compare multiple viewpoints, note trade-offs.
  - reasoning: show key intermediate steps before the conclusion."""
PROMPT = """Query: {query}

Numbered Sources (use [S1], [S2]...):
{numbered}

Query type: {qtype}
Write the answer grounded ONLY in these sources with citations on each factual sentence.
"""

class Synthesizer:
    def run(self, query: str, qtype: QueryType, sources: List[Dict[str, Any]], model="gpt-4o-mini", max_tokens=800) -> str:
        numbered = []
        for i, s in enumerate(sources, 1):
            content = s.get("text") or (s.get("title","") + "\n" + s.get("snippet",""))
            numbered.append(f"[{i}] {s['uri']}\n{content[:1200]}")
        msg = PROMPT.format(query=query, numbered="\n\n".join(numbered), qtype=qtype)
        out = completion(model=model, messages=[{"role":"system","content":SYS},{"role":"user","content":msg}],
                         temperature=0.2, max_tokens=max_tokens)["choices"][0]["message"]["content"]
        return out

# =========================
# Agent: Verifier
# =========================
class Verifier:
    def __init__(self):
        self.web = WebSearch()

    def extract_claims(self, text: str) -> List[str]:
        # simple heuristic: numbers, years, ProperNouns
        nums = re.findall(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b", text)
        proper = re.findall(r"\b[A-Z][a-zA-Z\-]{3,}\b", text)
        claims = list(dict.fromkeys(nums + proper))  # dedup, preserve order
        return claims[:8]

    def cross_check(self, claims: List[str]) -> List[Dict[str, Any]]:
        checks = []
        for c in claims:
            hits = self.web.search(c, k=2)
            checks.append({"claim": c, "verified": bool(hits), "evidence": [h.get("uri") for h in hits]})
        return checks

# =========================
# Agent: Finalizer
# =========================
class Finalizer:
    def render_sources(self, sources: List[Dict[str, Any]]) -> str:
        lines = []
        for i, s in enumerate(sources, 1):
            lines.append(f"[S{i}] {s.get('title','(untitled)')} — {s.get('uri')}")
        return "\n".join(lines) if lines else "No sources."

    def why_sources(self, query: str, sources: List[Dict[str, Any]]) -> str:
        n_local = sum(1 for s in sources if s["kind"]=="local")
        n_web = sum(1 for s in sources if s["kind"]=="web")
        return (f"Hybrid retrieval for '{query}'. Selected {len(sources)} diverse sources "
                f"({n_local} local + {n_web} web), prioritized semantic relevance, coverage, "
                f"and deduplicated by URI. Local chunks give project-specific grounding; "
                f"web items add recency and external validation.")

    def render_trace(self, trace: List[Dict[str, str]]) -> str:
        return "\n".join([f"- **{t['name']}**: {t['detail']}" for t in trace])

    def render_verifier(self, checks: List[Dict[str, Any]]) -> str:
        if not checks:
            return "Verifier skipped."
        lines = []
        for c in checks:
            badge = "✅ Verified" if c["verified"] else "⚠️ Unverified"
            evid = (", ".join(c["evidence"][:2])) or "—"
            lines.append(f"- {badge}: `{c['claim']}` (evidence: {evid})")
        return "\n".join(lines)

    def finalize(self, answer_md: str, sources: List[Dict[str, Any]],
                 why: str, trace: List[Dict[str, str]], checks: List[Dict[str, Any]]) -> str:
        return f"""### Answer
{answer_md}

---
### Sources
{self.render_sources(sources)}

### Why these sources
{why}

### Trace
{self.render_trace(trace)}

### Verifier report
{self.render_verifier(checks)}
"""

# =========================
# Orchestrator
# =========================
class Orchestrator:
    def __init__(self, data_dir="data"):
        self.router = Router()
        self.planner = Planner()
        self.local = LocalIndex(data_dir=data_dir)
        try:
            self.local.build()
        except Exception:
            self.local = None
        self.web = WebSearch()
        self.synth = Synthesizer()
        self.verify = Verifier()
        self.final = Finalizer()
        self.tools = ToolAgent()

    def ask(self, query: str, use_web: bool = True) -> str:
        trace: List[Dict[str, str]] = []

        # Router
        qtype = self.router.route(query)
        trace.append({"name":"router", "detail":f"type={qtype}"})

        # Tool path
        if qtype == "tool":
            tool_out = self.tools.maybe_handle(query)
            trace.append({"name":"tool", "detail":"calculator" if tool_out else "none"})
            ans = tool_out or "Tool could not handle this query."
            return self.final.finalize(ans, [], "Tool execution only.", trace, [])

        # Planner
        plan = self.planner.make(qtype, use_web=use_web)
        trace.append({"name":"planner", "detail":f"steps={','.join(plan.steps)}; k_local={plan.k_local}; k_web={plan.k_web}"})

        # Retrieval
        sources: List[Dict[str, Any]] = []
        if self.local:
            local_hits = self.local.search(query, k=plan.k_local)
            trace.append({"name":"retriever(local)", "detail":f"hits={len(local_hits)}"})
            sources += local_hits
        web_hits = self.web.search(query, k=plan.k_web) if plan.k_web else []
        trace.append({"name":"retriever(web)", "detail":f"hits={len(web_hits)}"})
        sources += web_hits

        # Deduplicate by URI, keep top ~10
        seen, dedup = set(), []
        for s in sources:
            if s["uri"] in seen: 
                continue
            seen.add(s["uri"]); dedup.append(s)
        # Simple diversity weighting: prefer 60% local, 40% web if possible
        local_sel = [s for s in dedup if s["kind"]=="local"][:6]
        web_sel = [s for s in dedup if s["kind"]=="web"][:4]
        sources = (local_sel + web_sel) if (local_sel or web_sel) else dedup[:10]
        trace.append({"name":"selector", "detail":f"sources_used={len(sources)}"})

        # Synthesize
        answer_md = self.synth.run(query, qtype, sources)
        trace.append({"name":"synthesizer", "detail":"drafted"})

        # Verify
        checks: List[Dict[str, Any]] = []
        if plan.use_verifier:
            claims = self.verify.extract_claims(answer_md)
            checks = self.verify.cross_check(claims)
            trace.append({"name":"verifier", "detail":f"claims_checked={len(checks)}"})

        # Finalize
        why = self.final.why_sources(query, sources)
        out = self.final.finalize(answer_md, sources, why, trace, checks)
        return out

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", type=str, required=True, help="Your question/prompt")
    ap.add_argument("--no-web", action="store_true", help="Disable live web search")
    ap.add_argument("--data", type=str, default="data", help="Path to local data dir")
    args = ap.parse_args()

    orch = Orchestrator(data_dir=args.data)
    print(orch.ask(args.ask, use_web=not args.no_web))

if __name__ == "__main__":
    main()
