import os
import json
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
LLM_MODEL = "gpt-4o-mini"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"

SYSTEM_PROMPT = """You are a clinical assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Do not speculate beyond what the sources state. Be concise and precise."""


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.index = None
        self.metadata: List[Dict] = []
        self._load_or_init_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _load_or_init_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
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

    def generate(self, query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "I could not find relevant information in the knowledge base to answer this question."

        context = "\n\n".join(
            f"[Source {i+1}: {c.get('source', 'unknown')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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