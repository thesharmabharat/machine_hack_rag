# Create README.md for the single-file RAG agent
readme = r"""# `rag_agent.py` — Single-File Agentic RAG

A **one-file, minimal Retrieval-Augmented Generation (RAG)** assistant that reads your local notes, searches the live web, and synthesizes a **source-grounded** answer with inline citations.

- ✅ Single Python file (`rag_agent.py`)
- 🔎 Local vector search (FAISS + sentence-transformers)
- 🌐 Web search (DuckDuckGo)
- 🧠 LLM synthesis via [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI/Azure/Anthropic/Groq compatible)
- 🧭 Simple query router: factual / open / reasoning
- 📎 Citations with a sources list

---

## 1) Quick Start

```bash
# 1) Create a virtual env (optional but recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install faiss-cpu sentence-transformers duckduckgo-search litellm

# 3) Put your local files in ./data (txt/md)
mkdir -p data && echo "My local note" > data/example.md

# 4) Configure your LLM key (example: OpenAI via LiteLLM)
export OPENAI_API_KEY=sk-...       # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
# (LiteLLM will pick the key; default model in the script is gpt-4o-mini)

# 5) Ask a question
python rag_agent.py --ask "Summarize my local notes about project X"
