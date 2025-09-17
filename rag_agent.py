# rag_agent.py (scoring-focused version)
import os, re
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
from litellm import completion

# ------------------------
# 1. Helpers
# ------------------------
def chunk_text(text, size=400, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ------------------------
# 2. Local Index
# ------------------------
class LocalIndex:
    def __init__(self, data_dir="data", model="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.model = SentenceTransformer(model)
        self.chunks, self.meta, self.index = [], [], None

    def build(self):
        texts, metas = [], []
        for p in self.data_dir.rglob("*"):
            if p.suffix.lower() in {".txt", ".md"}:
                raw = p.read_text(errors="ignore")
                for c in chunk_text(raw, size=400, overlap=100):
                    texts.append(c)
                    metas.append({"uri": str(p), "title": p.name, "snippet": c[:200]})
        if not texts: return
        embeds = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        index = faiss.IndexFlatIP(embeds.shape[1])
        index.add(np.array(embeds, dtype=np.float32))
        self.chunks, self.meta, self.index = texts, metas, index

    def search(self, query, k=8):
        if not self.index: return []
        q = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.array(q, dtype=np.float32), k*2)
        hits = []
        for sc, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            m = self.meta[idx]
            hits.append({**m, "id": f"L{idx}", "score": float(sc), "kind": "local", "text": self.chunks[idx]})
        q_emb = self.model.encode(query, normalize_embeddings=True)
        hits = sorted(hits, key=lambda h: util.cos_sim(q_emb, self.model.encode(h["text"], normalize_embeddings=True)).item(), reverse=True)
        return hits[:k]

# ------------------------
# 3. Web Search
# ------------------------
class WebSearch:
    def search(self, query, k=5):
        out = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=k)):
                out.append({
                    "id": f"W{i}",
                    "uri": r["href"],
                    "title": r["title"],
                    "snippet": r["body"][:200],
                    "score": 0.0,
                    "kind": "web",
                    "text": r["body"]
                })
        return out

# ------------------------
# 4. Router
# ------------------------
def classify_query(q: str):
    ql = q.lower()
    if any(x in ql for x in ["calculate", "sum", "add", "multiply"]):
        return "tool"
    if any(x in ql for x in ["why", "how", "derive", "step"]):
        return "reasoning"
    if len(ql.split()) <= 8 and q.endswith("?"):
        return "factual"
    return "open"

# ------------------------
# 5. Synthesizer
# ------------------------
def synthesize(query, query_type, sources, max_tokens=700):
    numbered = []
    for i, s in enumerate(sources, 1):
        text = s.get("text") or s["snippet"]
        numbered.append(f"[{i}] {s['uri']}\n{text[:800]}")
    sys = """You are a strict research assistant.
- Use ONLY the provided sources.
- EVERY claim must have a [S#] citation.
- If sources don’t contain the answer, say “Not enough info in sources.”
- Do NOT hallucinate or invent.
- Style:
  * factual → bullets first, then short explanation
  * open → synthesize multiple perspectives
  * reasoning → show steps before conclusion
"""
    user = f"Query: {query}\n\nSources:\n{chr(10).join(numbered)}\n\nQuery type: {query_type}."
    resp = completion(model="gpt-4o-mini", messages=[{"role":"system","content":sys},{"role":"user","content":user}], max_tokens=max_tokens)
    return resp["choices"][0]["message"]["content"]

# ------------------------
# 6. Verifier
# ------------------------
class Verifier:
    def __init__(self): self.web = WebSearch()
    def extract_claims(self, text):
        return list(set(re.findall(r"(\\b[A-Z][a-z]+\\b|\\d{4}|\\b\\d+\\.?\\d*\\b)", text)))[:5]
    def cross_check(self, claims):
        checks = []
        for c in claims:
            hits = self.web.search(c, k=2)
            checks.append((c, "Verified" if hits else "Unverified"))
        return checks

# ------------------------
# 7. Orchestrator
# ------------------------
def ask(query, use_web=True):
    trace = []
    qt = classify_query(query)
    trace.append(f"Router: query_type={qt}")

    if qt == "tool":
        try:
            result = eval(query.replace("calculate","").strip())
            return f"### Answer\nCalculation result: **{result}**\n\n---\n### Sources\nTool execution only.\n\n### Why these sources\nMath handled locally, no sources needed.\n\n### Trace\nRouter=tool → executed eval", []
        except: return "Error in calculation", []

    local = LocalIndex(); local.build()
    web = WebSearch()
    hits = local.search(query, k=6)
    if use_web: hits += web.search(query, k=5)
    trace.append(f"Retriever: local_hits={len(hits)}")

    seen, sources = set(), []
    for h in hits:
        if h["uri"] not in seen:
            seen.add(h["uri"]); sources.append(h)
    why = f"Selected {len(sources)} sources: {sum(1 for s in sources if s['kind']=='local')} local + {sum(1 for s in sources if s['kind']=='web')} web."

    answer = synthesize(query, qt, sources)
    trace.append("Synthesizer: answer drafted")

    if qt in {"factual","reasoning"}:
        v = Verifier(); claims = v.extract_claims(answer); checks = v.cross_check(claims)
        trace.append(f"Verifier: {checks}")

    src_str = "\n".join([f"[S{i+1}] {s['title']} — {s['uri']}" for i,s in enumerate(sources)])
    trace_str = "\n".join(trace)
    final = f"""### Answer
{answer}

---
### Sources
{src_str or "No sources"}

### Why these sources
{why}

### Trace
{trace_str}
"""
    return final, sources

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, required=True)
    parser.add_argument("--no-web", action="store_true")
    args = parser.parse_args()

    ans, srcs = ask(args.ask, use_web=not args.no_web)
    print(ans)
