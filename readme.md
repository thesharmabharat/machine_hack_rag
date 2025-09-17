# Agentic RAG – Dynamic Knowledge Companion

A production‑ready reference implementation of an **Agentic Retrieval‑Augmented Generation** system that adapts to query type and fuses **local knowledge**, **live web results**, and **optional external tools/APIs**. The assistant behaves like a **dynamic knowledge companion**: plans, retrieves, reasons, verifies, and cites.

---

## ✨ Highlights

* **Multi‑agent architecture**: Router → Planner → Retriever(s) → Synthesizer → Verifier → Finalizer.
* **Hybrid retrieval**: FAISS over your local docs + live web search (DuckDuckGo, Tavily, or SerpAPI).
* **Dynamic strategies** by query type: factual, exploratory, reasoning‑heavy, and data/tool tasks.
* **Transparent outputs**: citations, source rationale, and a trace of the reasoning steps.
* **Low‑latency defaults** with optional re‑ranking and cross‑validation passes.

---

## 🗂️ Repo Layout

```
agentic-rag/
├─ README.md                        # This file
├─ requirements.txt
├─ .env.example
├─ data/
│  └─ sample_docs/                  # Put your PDFs/CSVs/TXTs/MDs here
├─ app/
│  ├─ main.py                       # CLI & simple REST server
│  ├─ config.py                     # Settings & feature flags
│  ├─ schema.py                     # Pydantic models for I/O contracts
│  ├─ orchestrator.py               # High-level pipeline orchestration
│  ├─ agents/
│  │  ├─ router.py                  # Query type classifier
│  │  ├─ planner.py                 # Step planner
│  │  ├─ retriever.py               # Multi-source retrieval coordinator
│  │  ├─ synthesizer.py             # Answer generation with citations
│  │  ├─ verifier.py                # Cross-check critical claims
│  │  └─ finalizer.py               # UX formatting & transparency bundle
│  ├─ retrievers/
│  │  ├─ local_index.py             # FAISS + sentence-transformers
│  │  ├─ web_search.py              # DuckDuckGo/Tavily/SerpAPI adapters
│  │  └─ reranker.py                # (Optional) Cross-encoder reranker
│  ├─ tools/
│  │  ├─ calculator.py              # Safe eval math tool
│  │  ├─ code_exec_stub.py          # (Optional) sandboxes for code
│  │  └─ example_api.py             # (Optional) external API helper
│  └─ utils/
│     ├─ io.py                      # Loader for PDF, MD, TXT, CSV
│     ├─ text.py                    # Chunking, cleaning
│     ├─ llm.py                     # LLM abstraction (OpenAI/LiteLLM)
│     ├─ citations.py               # Citation formatting & de-dup
│     └─ trace.py                   # Structured logging
└─ tests/
   └─ smoke_tests.py                # Minimal e2e validation
```

---

## 🔧 Setup

1. **Python**: 3.10+
2. **Install**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Environment**: copy & edit `.env.example` → `.env` (put keys if you’ll use LLM APIs or web providers):

   ```env
   OPENAI_API_KEY=...
   LLM_PROVIDER=openai   # or azure_openai, groq, anthropic via litellm
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

   # Web search (pick any):
   TAVILY_API_KEY=...
   SERPAPI_API_KEY=...
   SERPER_API_KEY=...
   WEB_PROVIDER=duckduckgo  # duckduckgo | tavily | serpapi | serper

   # Optional: Reranker
   RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   ```
4. **Index local docs** (place files in `data/sample_docs/`), then:

   ```bash
   python -m app.main --reindex
   ```

---

## ▶️ Run

### A) CLI

```bash
python -m app.main --ask "What are the key risks mentioned in the quarterly PDF?"
```

### B) Simple REST (FastAPI)

```bash
python -m app.main --serve --host 0.0.0.0 --port 8000
```

Then POST to `/ask` with JSON:

```json
{
  "query": "Summarize the latest findings on X and contrast with our local report",
  "options": {"use_web": true, "max_tokens": 800}
}
```

---

## 🧠 Agent Workflow (at a glance)

1. **Router**: classify query → `{factual|open|reasoning|tool}` and select strategy knobs (k, web\_on, depth).
2. **Planner**: create a short plan (steps & tools).
3. **Retriever**: parallel local index search + web search; optional re-ranking; deduplicate; normalize citations.
4. **Synthesizer**: instruct LLM to **ground generations** on retrieved chunks; produce draft with inline cite markers `[S1]`…
5. **Verifier**: extract critical claims → corroborate via second-pass web checks; flag inconsistencies → feedback loop.
6. **Finalizer**: compose answer + **Sources**, **Why these sources**, **Trace**.

---

## ✅ Evaluation (quick smoke tests)

```bash
pytest -q
```

* **Answer quality**: basic assertions (has citations, no empty sections).
* **Adaptability**: router labels & plan contain expected stages.
* **Grounding**: every claim block references a retrieved source id.

---

## 📁 Code

### requirements.txt

```txt
fastapi==0.114.2
uvicorn==0.30.6
pydantic==2.9.2
python-dotenv==1.0.1
faiss-cpu==1.8.0.post1
sentence-transformers==3.0.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
duckduckgo-search==6.1.7
httpx==0.27.2
rich==13.8.1
orjson==3.10.7
jinja2==3.1.4
# Optional LLMs (choose one path):
litellm==1.48.1   # universal LLM adapter (OpenAI/Anthropic/Groq/Azure)
```