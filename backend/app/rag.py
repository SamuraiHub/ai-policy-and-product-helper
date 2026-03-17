import sys, logging
import time, os, math, json, hashlib, re
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm
import uuid
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

# ---- Simple local embedder (deterministic) ----
def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.split()]

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        points = []
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            point_id = str(uuid.UUID(h[:32]))

            points.append(qm.PointStruct(
                id=point_id,  # ✅ deterministic ID
                vector=v.tolist(),
                payload=m
            ))

        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"""
        You are a strict policy assistant.

        RULES:
        - ONLY use exact information from sources
        - DO NOT rephrase into new meanings
        - Prefer quoting or closely paraphrasing source wording
        - You may combine multiple sources
        - Avoid introducing new terms not present in sources
        - DO NOT include citations in the answer
        - Citations will be added separately

        Question:
        {query}

        Sources:
        """

        prompt += "\n".join([
            f"{c['title']} | {c.get('section')}\n{c['text'][:400]}"
            for c in contexts
        ])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
    def __init__(self):
        self.embedder = LocalEmbedder()
        # Vector store selection
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384)
            except Exception:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        # LLM selection
        if settings.llm_provider == "openrouter" and settings.openrouter_api_key:
            try:
                logger.debug("Initializing OpenRouterLLM...")
                self.llm = OpenRouterLLM(api_key=settings.openrouter_api_key, model=settings.llm_model)
                self.llm_name = f"openrouter:{settings.llm_model}"
                logger.debug("OpenRouterLLM initialized successfully.")
            except Exception as e:
                logger.exception("OpenRouterLLM failed")  # logs full stack trace
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text_norm = normalize_text(ch["text"])
            h = hashlib.sha256(text_norm.encode("utf-8")).hexdigest()
            point_id = str(uuid.UUID(h[:32])) 

            meta = {
                "id": point_id,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text_norm,
            }

            v = self.embedder.embed(text_norm)
            vectors.append(v)
            metas.append(meta)

            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))
    
    def is_relevant(self, m, query):
        q = query.lower()
        text = (m.get("title", "") + " " + m.get("text", "")).lower()

        # ❌ Block obvious noise
        if "catalog" in m.get("title", "").lower():
            return False

        query_terms = set(q.split())

        overlap = sum(1 for t in query_terms if t in text)

        # 🔥 Require BOTH:
        # 1. keyword overlap
        # 2. semantic relevance (from vector score)
        return overlap >= 2 and m.get("_score", 0) > 0.3
    
    def expand_query(self, q: str) -> str:
        synonyms = {
            "damaged": "defective broken faulty",
            "return": "refund exchange",
            "shipping": "delivery sla",
        }

        q_lower = q.lower()
        extra = []

        for k, v in synonyms.items():
            if k in q_lower:
                extra.append(v)

        return q + " " + " ".join(extra)

    def retrieve(self, query: str, k: int = 40) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(self.expand_query(query))

        results = self.store.search(qv, k=k)

        # ✅ Step 1: deduplicate by (title + section)
        seen = set()
        unique = []

        for score, meta in results:
            key = (meta.get("title"), meta.get("section"))
            if key in seen:
                continue
            seen.add(key)
            meta["_score"] = score
            unique.append(meta)

        # ✅ Step 2: simple keyword boost (hybrid-lite)
        query_terms = set(query.lower().split())
        for m in unique:
            text = m.get("text", "").lower()
            title = m.get("title", "").lower()

            overlap = sum(1 for t in query_terms if t in text)

            # 🔥 NEW: policy boost
            policy_boost = 0
            if any(k in title for k in ["policy", "refund", "warranty", "shipping"]):
                policy_boost = 0.2

            m["_final"] = (m["_score"] * 0.6) + (overlap * 0.3) + policy_boost

        ranked = sorted(unique, key=lambda x: x["_final"], reverse=True)

        self.metrics.add_retrieval((time.time()-t0)*1000.0)

        filtered = [m for m in ranked if self.is_relevant(m, query)]

        return filtered[:4]
    
    def dedupe_for_llm(self, contexts):
        seen = set()
        out = []

        for c in contexts:
            key = (c["title"], c.get("section"))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)

        return out

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        contexts = self.dedupe_for_llm(contexts)
        # Always keep top 2
        top = contexts[:2]

        # Filter the rest
        rest = [
            c for c in contexts[2:]
            if c.get("_score", 0) > 0.35
        ]

        contexts = top + rest
        answer = self.llm.generate(query, contexts)
        self.metrics.add_generation((time.time()-t0)*1000.0)
        return answer

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m
        }

# ---- Helpers ----
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def build_chunks_from_docs(docs, chunk_size, overlap):
    out = []
    seen_hashes = set()
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            ch_norm = normalize_text(ch)
            h = hashlib.sha256(ch_norm.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            out.append({"title": d["title"], "section": d["section"], "text": ch_norm})
            seen_hashes.add(h)
    return out
