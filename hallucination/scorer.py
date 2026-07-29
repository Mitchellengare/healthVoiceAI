import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from typing import List, Dict
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForNextSentencePrediction
import torch

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NSP_MODEL = "bert-base-uncased"
USE_NSP = True

THRESHOLDS = {"LOW": 0.35, "HIGH": 0.65}  # hallucination score boundaries


class HallucinationScorer:
    def __init__(self):
        print("[Hallucination] Loading NLI model...")
        self.nli = pipeline("text-classification", model=NLI_MODEL, device=-1)

        if USE_NSP:
            print("[Hallucination] Loading NSP model...")
            self.nsp_tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL)
            self.nsp_model = AutoModelForNextSentencePrediction.from_pretrained(NSP_MODEL)
            self.nsp_model.eval()

    def score(self, answer: str, chunks: List[Dict]) -> Dict:
        if not chunks or not answer:
            return {"score": 0.0, "label": "LOW", "flagged_sentences": []}

        sentences = self._split_sentences(answer)
        context = " ".join(c["text"] for c in chunks)

        sentence_scores = []
        flagged = []

        for sent in sentences:
            nli_score = self._nli_score(sent, context)
            nsp_score = self._nsp_score(context, sent) if USE_NSP else 0.5

            # NLI is primary, NSP is secondary signal
            combined = 0.7 * nli_score + 0.3 * nsp_score
            sentence_scores.append(combined)
            if combined > THRESHOLDS["LOW"]:
                flagged.append({"sentence": sent, "score": round(combined, 3)})

        overall = float(np.mean(sentence_scores)) if sentence_scores else 0.0
        label = self._label(overall)

        return {
            "score": round(overall, 3),
            "label": label,
            "flagged_sentences": flagged,
        }

    def _nli_score(self, hypothesis: str, premise: str) -> float:
        """
        Returns hallucination probability (0=entailed/faithful, 1=contradicted/hallucinated).
        NLI labels: ENTAILMENT -> faithful, CONTRADICTION -> hallucinated, NEUTRAL -> uncertain.
        """
        result = self.nli(
            f"{premise[:512]} [SEP] {hypothesis}",
            truncation=True,
            max_length=512,
        )
        label = result[0]["label"].upper()
        conf = result[0]["score"]

        if label == "ENTAILMENT":
            return 1.0 - conf
        elif label == "CONTRADICTION":
            return conf
        else:
            return 0.5

    def _nsp_score(self, context: str, answer_sentence: str) -> float:
        """
        BERT NSP: P(answer_sentence follows logically from context).
        High NSP probability -> answer is coherent with source -> lower hallucination risk.
        """
        try:
            inputs = self.nsp_tokenizer(
                context[:256],
                answer_sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = self.nsp_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            # NSP: label 0 = IsNext, label 1 = NotNext
            is_next_prob = probs[0][0].item()
            # Convert: high is_next -> low hallucination
            return 1.0 - is_next_prob
        except Exception:
            return 0.5

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter — swap with spaCy for production."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _label(self, score: float) -> str:
        if score < THRESHOLDS["LOW"]:
            return "LOW"
        elif score < THRESHOLDS["HIGH"]:
            return "MEDIUM"
        else:
            return "HIGH"