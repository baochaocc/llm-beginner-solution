"""任务四（加分 S2）：bge-reranker-base 精排。

reranker 吃 [query, doc] 文本对打分（不是再算一遍 embedding）。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Reranker:
    def __init__(self, model_dir=None, device="cuda"):
        self.device = device
        self.model_dir = model_dir or (Path(__file__).resolve().parents[1] / "models" / "bge-reranker-base")
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)).to(self.device).eval()
            self._torch = torch

    def rerank(self, query, docs, k):
        self._ensure()
        torch = self._torch
        pairs = [(query, d["text"][:1000]) for d in docs]
        enc = self._tok(pairs, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self.device)
        with torch.no_grad():
            scores = self._model(**enc).logits.squeeze(-1)   # (B,)
        order = scores.argsort(descending=True).cpu().tolist()
        out = []
        for j in order:
            d = dict(docs[j])
            d["score"] = float(scores[j].item())
            out.append(d)
        return out[:k]
