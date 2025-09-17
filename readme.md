# Agentic RAG – Single File Implementation

A **one-file, multi-agent Retrieval-Augmented Generation (RAG)** pipeline that dynamically adapts to different query types and behaves like a **knowledge companion** instead of a static Q&A bot.

---

## ✨ Features

- **Multi-Agent Workflow**  
  Router → Planner → Retriever (Local + Web) → Synthesizer → Verifier → Finalizer  

- **Hybrid Retrieval**  
  - Local knowledge base (`.txt` / `.md` in `./data/`)  
  - Live web information via DuckDuckGo  
  - Deduplication + reranking for precision  

- **Dynamic Query Handling**  
  - **Factual** → concise, trusted sources  
  - **Open-ended** → synthesize multiple perspectives  
  - **Reasoning-heavy** → show intermediate steps  
  - **Tool** → calculator for numeric queries  

- **Transparency & Trust**  
  - Inline `[S#]` citations for every fact  
  - **Why these sources** section explaining retrieval  
  - **Trace** of agent steps (router, retriever, synthesizer, verifier)  
  - **Verifier report**: cross-checks claims against live web  

---

## 📂 Structure

All logic is in **one file**: `agentic_rag.py`  

