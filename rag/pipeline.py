import os
import json

# faiss and torch each bundle OpenMP; multithreading both segfaults on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langsmith.wrappers import wrap_openai

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for macOS
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
LLM_MODEL = "gpt-4o-mini"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"

SYSTEM_PROMPT = """You are a clinical assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Do not speculate beyond what the sources state. Be concise and precise."""


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = wrap_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        self.index: faiss.Index = faiss.IndexFlatIP(EMBED_DIM)
        self.metadata: List[Dict] = []
        self._load_or_init_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _load_or_init_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            if self.index.d != EMBED_DIM:
                raise RuntimeError(
                    f"Index dimension {self.index.d} != {EMBED_DIM} for {EMBED_MODEL}. "
                    "Rebuild it with: python -m utils.reindex"
                )
            with open(META_PATH) as f:
                self.metadata = json.load(f)
            print(f"[RAG] Loaded index with {len(self.metadata)} chunks.")
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            print("[RAG] Fresh index initialized.")

    def _save_index(self):
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.metadata, f)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, texts: List) -> int:
        """
        texts: list of strings OR dicts with {"text": ..., "source": ...}
        Returns number of chunks added.
        """
        records = []
        for t in texts:
            if isinstance(t, str):
                records.append({"text": t, "source": "unknown"})
            else:
                records.append(t)

        raw_texts = [r["text"] for r in records]
        embeddings = self._embed(raw_texts)
        self.index.add(embeddings)
        self.metadata.extend(records)
        self._save_index()
        return len(records)

    def rebuild(self, records: List[Dict]) -> int:
        """Re-embed `records` into a fresh index (use after changing the embedding model)."""
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.metadata = []
        if records:
            self.index.add(self._embed([r["text"] for r in records]))
            self.metadata = list(records)
        self._save_index()
        return len(self.metadata)

    def replace_sources(self, records: List[Dict]) -> int:
        """Add `records`, first dropping existing chunks from the same sources (no duplicates)."""
        sources = {r["source"] for r in records}
        kept = [m for m in self.metadata if m.get("source") not in sources]
        self.rebuild(kept + list(records))
        return len(records)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        q_emb = self._embed([query])
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["relevance_score"] = float(score)
            results.append(chunk)
        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: List[Dict], language: Optional[str] = None) -> str:
        if not chunks:
            return "I could not find relevant information in the knowledge base to answer this question."

        context = "\n\n".join(
            f"[Source {i+1}: {c.get('source', 'unknown')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        system = SYSTEM_PROMPT
        if language:
            system += f"\nWrite your entire answer in {language}."
        else:
            system += "\nAnswer in the same language as the question."

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype(np.float32)