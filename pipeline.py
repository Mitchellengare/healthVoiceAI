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
META_PATH = "data/meta/.json"

SYSTEM_PROMPT = """ You are a clinical assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Do not speculate beyond what the sources state. Be concise and precise."""

class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = OpenAI