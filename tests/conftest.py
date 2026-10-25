import sys, types
from unittest.mock import MagicMock
import pytest

ROOT = __file__.rsplit("/tests/", 1)[0]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def appmod(monkeypatch):
    """Import app with the heavy RAG/scorer modules stubbed out."""
    rag_mod = types.ModuleType("rag.pipeline")
    rag_mod.RAGPipeline = MagicMock()
    scorer_mod = types.ModuleType("hallucination.scorer")
    scorer_mod.HallucinationScorer = MagicMock()
    monkeypatch.setitem(sys.modules, "rag.pipeline", rag_mod)
    monkeypatch.setitem(sys.modules, "hallucination.scorer", scorer_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    sys.modules.pop("app", None)
    import app
    app.USER_LANG_PREF.clear()
    yield app
    sys.modules.pop("app", None)
