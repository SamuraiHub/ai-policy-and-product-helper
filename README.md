# 🧠 AI Policy & Product Helper

A Retrieval-Augmented Generation (RAG) system that answers company policy and product questions using grounded, cited knowledge from internal documents.

---

# 🚀 Features

* 🔍 Semantic search over company documents (policies, shipping, warranty, etc.)
* 🧠 LLM-powered answers (OpenRouter or stub fallback)
* 📚 Source-grounded responses with citations
* 🧩 Chunk-level explainability (view supporting text)
* ⚡ Hybrid retrieval (vector + keyword boosting)
* 🐳 Fully containerized (Docker Compose)
* 🧪 Unit-testable (no external dependencies required)

---

## 🏗️ Architecture

```text
User (Next.js UI)
   ↓
FastAPI Backend (/api/ask)
   ↓
RAG Engine
   ├─ Query Expansion
   ├─ Embedding (MiniLM)
   ├─ Retrieval (Qdrant / Memory)
   ├─ Hybrid Ranking (vector + keyword)
   ├─ Filtering + Deduplication
   ↓
LLM (OpenRouter / Stub)
   ↓
Answer + Citations + Supporting Chunks
```

---

## 🔹 Components

### 1. Frontend (Next.js)

* Chat interface
* Displays:

  * Answer
  * Citations (badges)
  * Expandable supporting chunks

---

### 2. Backend (FastAPI)

Core endpoints:

* `POST /api/ingest` → Load & index documents
* `POST /api/ask` → Ask questions
* `GET /api/metrics` → Performance stats
* `GET /api/health` → Health check

---

### 3. RAG Engine (`rag.py`)

Pipeline:

1. **Query Expansion**

   * Adds synonyms (e.g., "damaged → defective")

2. **Embedding**

   * Model: `all-MiniLM-L6-v2`

3. **Retrieval**

   * Vector similarity (cosine)
   * Keyword overlap boost
   * Policy-aware boosting

4. **Filtering**

   * Remove irrelevant docs (e.g., product catalog)
   * Require semantic + keyword match

5. **Deduplication**

   * By `(title, section)`

6. **LLM Generation**

   * Strict prompt:

     * No hallucination
     * Grounded in sources
     * No inline citations (handled separately)

---

### 4. Vector Store

| Option   | Description                    |
| -------- | ------------------------------ |
| Qdrant   | Persistent, scalable vector DB |
| InMemory | Lightweight, used for tests    |

---

### 5. LLM Layer

| Provider   | Usage              |
| ---------- | ------------------ |
| OpenRouter | Production         |
| Stub       | Testing / fallback |

---

# ⚙️ Setup

## 1. Clone & Run

```bash
docker compose up --build
```

---

## 2. Ingest Documents

* Open frontend → Admin tab
* Click **Ingest**

OR via API:

```bash
curl -X POST http://localhost:8000/api/ingest
```

---

## 3. Ask Questions

Example:

```text
Can a customer return a damaged blender after 20 days?
```

---

## 4. 🔐 Environment Variables

- Copy `.env.example` → `.env`
- Do NOT commit `.env` (contains API keys)

---

# 🧪 Running Tests

## ✅ Recommended (fast, no dependencies)

```bash
docker compose run --no-deps --rm backend pytest -q 
```
# results:
..  [100%]
2 passed in 20.23s
---

## 🔍 What is tested

* API health
* Document ingestion
* Retrieval + answer generation
* Citation presence

---

# 📂 Project Structure

```
backend/
  app/
    main.py          # FastAPI app
    rag.py           # RAG engine
    ingest.py        # Document loading & chunking
    models.py        # API schemas
    settings.py      # Config
    tests/           # Unit tests

data/
  *.md               # Policy & product docs

frontend/
  Next.js app
```

---

# 🧠 Key Design Decisions

## 1. Hybrid Retrieval (Vector + Keyword)

**Why:**

* Pure embeddings miss exact matches
* Keywords catch critical terms

**Trade-off:**

* Slightly more complexity
* Much better accuracy

---

## 2. Strict LLM Prompting

**Goal:**

* Eliminate hallucination

**Rules:**

* Only use provided sources
* No external knowledge
* No invented phrasing

**Trade-off:**

* Slightly less “fluent” answers
* Much more reliable

---

## 3. Deduplication at Multiple Layers

* Retrieval level
* LLM input level
* UI level

**Why:**

* Prevent repeated chunks
* Cleaner UX

---

## 4. In-Memory Mode for Tests

**Why:**

* No dependency on Qdrant
* Faster (<1s tests)

---

## 5. Chunking Strategy

* Fixed size + overlap

**Trade-off:**

* Simple & fast
* Not semantically perfect

---

# ⚠️ Known Limitations

* No cross-encoder reranking (yet)
* Chunking is not semantic-aware
* No caching layer
* No streaming responses
* LLM latency depends on OpenRouter

---

# 🚀 What I Would Ship Next

## 🔥 1. Cross-Encoder Reranker (BIGGEST WIN)

* Re-rank top results using transformer
* Improves accuracy significantly

---

## 🔥 2. Hybrid Search in Qdrant (BM25 + Vector)

* True hybrid retrieval (not manual boosting)

---

## 🔥 3. Streaming Responses

* Better UX (ChatGPT-like typing)

---

## 🔥 4. Answer Confidence Score

* Show reliability of response

---

## 🔥 4. Evaluation pipeline

* Automated accuracy testing on known queries

---

## 🔥 5. Caching Layer

* Cache embeddings + answers
* Reduce latency & cost

---

## 🔥 6. Semantic Chunking

* Split by meaning instead of length

---

## 🔥 7. Observability

* Logging:

  * retrieved chunks
  * scores
  * LLM latency

---

## 🔥 8. CI/CD Pipeline

* GitHub Actions:

  * run tests
  * lint
  * build images

---

# 📊 Example Output

### Question

```
What’s the shipping SLA to East Malaysia for bulky items?
```

### Answer

```
7–10 business days with a surcharge.
```

### Citations

```
Delivery_and_Shipping.md | SLA
```

---

# 🏁 Summary

This system demonstrates:

* End-to-end RAG pipeline
* Grounded, explainable answers
* Clean separation of concerns
* Production-ready architecture

---

# 🙌 Muaz Ajarhi

Built as a practical, production-oriented RAG system for policy and product knowledge.

