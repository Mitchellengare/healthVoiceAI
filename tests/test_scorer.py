import sys, types
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def scorer(monkeypatch):
    # torch/transformers are heavy; stub them and test our logic only.
    tf = types.ModuleType("transformers")
    tf.pipeline = MagicMock()
    tf.AutoTokenizer = MagicMock()
    tf.AutoModelForNextSentencePrediction = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    sys.modules.pop("hallucination.scorer", None)
    from hallucination.scorer import HallucinationScorer
    s = HallucinationScorer()
    yield s
    sys.modules.pop("hallucination.scorer", None)


def test_nli_gets_text_pair(scorer):
    scorer.nli = MagicMock(return_value={"label": "entailment", "score": 0.9})
    assert scorer._nli_score("hyp", "prem") == pytest.approx(0.1)
    assert scorer.nli.call_args.args[0] == {"text": "prem", "text_pair": "hyp"}


def test_contradiction_and_neutral(scorer):
    scorer.nli = MagicMock(return_value={"label": "CONTRADICTION", "score": 0.8})
    assert scorer._nli_score("h", "p") == 0.8
    scorer.nli = MagicMock(return_value=[{"label": "neutral", "score": 0.8}])
    assert scorer._nli_score("h", "p") == 0.5


def test_labels(scorer):
    assert [scorer._label(x) for x in (0.1, 0.5, 0.9)] == ["LOW", "MEDIUM", "HIGH"]


def test_empty_inputs(scorer):
    assert scorer.score("", [{"text": "x"}])["label"] == "LOW"
    assert scorer.score("some answer here", [])["label"] == "LOW"


def test_score_flags_hallucinated(scorer):
    scorer.nli = MagicMock(return_value=[{"label": "CONTRADICTION", "score": 1.0}])
    r = scorer.score("This sentence is long enough.", [{"text": "ctx"}])
    assert r["label"] in ("MEDIUM", "HIGH") and r["flagged_sentences"]
