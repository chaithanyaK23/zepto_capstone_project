# Zepto Support Assistant

A Retrieval-Augmented Generation (RAG) pipeline that answers Zepto customer-support queries using local policy documents, ChromaDB vector search, a LangGraph workflow, and a FastAPI endpoint.

---

## Architecture Flow

```
+---------------------------------------------------------------------+
|                        OFFLINE  (run once)                          |
|                                                                     |
|   docs/doc_01.txt ... doc_08.txt                                    |
|           |                                                         |
|           v                                                         |
|   1. INGESTION  --  load_documents()                                |
|           |         Reads all 8 policy .txt files from docs/        |
|           v                                                         |
|   2. EMBEDDING  --  embed_texts()                                   |
|           |         sentence-transformers/all-MiniLM-L6-v2          |
|           |         produces normalised 384-dim float vectors        |
|           v                                                         |
|      ChromaDB  (chroma_store/, cosine similarity)                   |
|           |         Persists ids, documents, metadata, embeddings   |
+-----------|-------------------------------------------------------------+
            |
+-----------|----------------------------------------------------- ONLINE -+
|           v                                                         |
|   POST /ask  { "query": "..." }                                     |
|           |                                                         |
|           v                                                         |
|   LangGraph  -- classify_intent node                                |
|           |     keyword heuristic (or optional Groq LLM)            |
|           |                                                         |
|      +----+------------------------------+                          |
|      | policy_question                   | general_question         |
|      v                                   v                          |
|  3. RETRIEVAL                     direct_answer node                |
|     query_collection()            Returns canned / LLM reply        |
|     top-3 cosine matches                                            |
|      |                                                              |
|      v                                                              |
|  4. GENERATION                                                      |
|     Mock mode  -> deterministic snippet from top chunk              |
|     Real mode  -> Groq LLM call with structured JSON prompt         |
|      |           (up to 3 retries with Pydantic validation)         |
|      v                                                              |
|   AskResponse { answer, sources, confidence }                       |
+---------------------------------------------------------------------+
```

### Stage Details

| Stage | File | Key Function |
|---|---|---|
| **Ingestion** | `ingest.py` | `load_documents()` — reads & validates exactly 8 docs |
| **Embedding** | `ingest.py` | `embed_texts()` — MiniLM model, L2-normalised |
| **Storage** | `ingest.py` | `get_chroma_collection()` — persistent cosine-space collection |
| **Routing** | `rag_graph.py` | `classify_intent` node — keyword or Groq LLM |
| **Retrieval** | `ingest.py` | `query_collection()` — top-3 nearest neighbours |
| **Generation** | `rag_graph.py` | `retrieve_and_answer` / `direct_answer` nodes |
| **Schema** | `schemas.py` | `AskResponse` — Pydantic model with JSON validation |
| **API** | `main.py` | FastAPI `POST /ask` endpoint |

---

## Project Structure

```
support-assistant/
+-- docs/                    # 8 Zepto policy documents (source corpus)
|   +-- doc_01.txt           # Delivery policy
|   +-- doc_02.txt           # Return policy
|   +-- doc_03.txt           # Refund policy
|   +-- doc_04.txt           # Membership / Zepto Pass
|   +-- doc_05.txt           # Order tracking
|   +-- doc_06.txt           # Cancellation policy
|   +-- doc_07.txt           # Gift cards
|   +-- doc_08.txt           # Customer support hours
+-- chroma_store/            # Auto-created -- ChromaDB persistence directory
+-- ingest.py                # Ingestion, embedding, and retrieval helpers
+-- rag_graph.py             # LangGraph workflow (classify -> retrieve -> generate)
+-- prompt_template.py       # Structured few-shot prompt for real-LLM mode
+-- schemas.py               # Pydantic request / response models
+-- main.py                  # FastAPI application
+-- demo_requests.py         # Quick smoke-test: one policy + one general query
+-- Dockerfile               # Container definition
+-- requirements.txt         # Python dependencies
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Build the vector index (one-time)

```bash
python ingest.py
# Embedded and stored 8 Zepto policy chunks in ChromaDB.
```

> The index is rebuilt automatically on first API startup if `chroma_store/` is missing or incomplete.

### 3 — Start the API

```bash
uvicorn main:app --reload
# Uvicorn running on http://127.0.0.1:8000
```

### 4 — Send a query

**Policy query** (triggers retrieval path):

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Zepto delivery fee for small orders?"}'
```

Expected response shape:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
```

**General / unrelated query** (skips retrieval):

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who won the football match yesterday?"}'
```

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Query Routing

Routing is performed by the `classify_intent` LangGraph node.

| Condition | Intent | Next Node |
|---|---|---|
| Query contains a policy keyword | `policy_question` | `retrieve_and_answer` |
| Query contains no policy keyword | `general_question` | `direct_answer` |

**Policy keywords:** `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`

---

## Response Schema

Every response is validated against `AskResponse` (Pydantic):

```json
{
  "answer":     "assistant answer string (up to 120 words in real-LLM mode)",
  "sources":    ["doc_01", "doc_02"],
  "confidence": 0.95
}
```

| Field | Type | Constraint |
|---|---|---|
| `answer` | `str` | Required, non-empty |
| `sources` | `list[str]` | Chunk/document IDs; empty for general queries |
| `confidence` | `float` | `0.0 to 1.0` inclusive |

---

## Mock vs Real-LLM Mode

| Mode | `MOCK_LLM` env var | Behaviour |
|---|---|---|
| **Mock** (default) | `1` or unset | Deterministic answers; no external API calls |
| **Real LLM** | `0` | Calls Groq API; requires `GROQ_API_KEY` |

To enable real-LLM mode:

```bash
export MOCK_LLM=0
export GROQ_API_KEY=gsk_...
export GROQ_MODEL=llama-3.1-8b-instant   # optional, this is the default
uvicorn main:app --reload
```

---

## Running Demo Requests

```bash
python demo_requests.py
```

Fires one policy query and one general query against an in-process `TestClient` and prints the raw JSON responses.

---

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 8000:8000 zepto-support-assistant
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | HTTP API framework |
| `uvicorn` | ASGI server |
| `langgraph` | Stateful graph orchestration |
| `chromadb` | Local vector database |
| `sentence-transformers` | Local embedding model (MiniLM-L6-v2) |
| `pydantic` | Request/response validation |
| `requests` | Optional Groq API calls |
