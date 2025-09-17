# rag_agent.py
import os, re, requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from duckduckgo_search import DDGS
from litellm import completion

# ------------------------
# 1. Local Index
# ------------------------
class LocalIndex:
    def __init__(self, data_dir="data", model="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.model = SentenceTransformer(model)
        self.docs, self.meta, self.index = [], [], None

    def build(self):
        texts, metas = [], []
        for p in self.data_dir.rglob("*"):
            if p.suffix.lower() in {".txt", ".md"}:
                text = p.read_text(errors="ignore")[:3000]
                texts.append(text)
                metas.append({"uri": str(p), "title": p.name, "snippet": text[:300]})
        embeds = self.model.encode(texts, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeds.shape[1])
        index.add(np.array(embeds, dtype=np.float32))
        self.docs, self.meta, self.index = texts, metas, index

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.array(q, dtype=np.float32), k)
        out = []
        for sc, idx in zip(scores[0], ids[0]):
            m = self.meta[idx]
            out.append({**m, "id": f"L{idx}", "score": float(sc), "kind": "local", "text": self.docs[idx]})
        return out

# ------------------------
# 2. Web Search
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
# 3. Simple Router
# ------------------------
def classify_query(q: str):
    ql = q.lower()
    if any(x in ql for x in ["why", "how", "derive", "step"]):
        return "reasoning"
    if len(ql.split()) <= 8 and q.endswith("?"):
        return "factual"
    return "open"

# ------------------------
# 4. Synthesizer
# ------------------------
def synthesize(query, query_type, sources, max_tokens=600):
    numbered = []
    for i, s in enumerate(sources, 1):
        text = s.get("text") or s["snippet"]
        numbered.append(f"[{i}] {s['uri']}\n{text[:800]}")
    sys = "You are a careful assistant. Only use provided sources. Cite with [S1], [S2]."
    user = f"Query: {query}\n\nSources:\n{chr(10).join(numbered)}\n\nAnswer grounded in sources. Query type: {query_type}."
    resp = completion(model="gpt-4o-mini", messages=[{"role":"system","content":sys},{"role":"user","content":user}], max_tokens=max_tokens)
    return resp["choices"][0]["message"]["content"]

# ------------------------
# 5. Orchestrator
# ------------------------
def ask(query, use_web=True):
    query_type = classify_query(query)
    local = LocalIndex(); local.build()
    web = WebSearch()

    hits = local.search(query, k=5)
    if use_web:
        hits += web.search(query, k=5)

    # Dedup
    seen, sources = set(), []
    for h in hits:
        if h["uri"] not in seen:
            seen.add(h["uri"]); sources.append(h)
    answer = synthesize(query, query_type, sources)
    return answer, sources

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
    print("### Answer\n", ans, "\n\n---\n### Sources")
    for i, s in enumerate(srcs, 1):
        print(f"[S{i}] {s['title']} — {s['uri']}")
