"""Rebuild the FAISS index from data/meta.json using the current embedding model."""
import json

from rag.pipeline import RAGPipeline, EMBED_MODEL, META_PATH


def main():
    with open(META_PATH) as f:
        records = json.load(f)
    rag = RAGPipeline.__new__(RAGPipeline)  # skip loading the stale index
    from sentence_transformers import SentenceTransformer
    rag.embedder = SentenceTransformer(EMBED_MODEL)
    print(f"Reindexed {rag.rebuild(records)} chunks with {EMBED_MODEL}.")


if __name__ == "__main__":
    main()
