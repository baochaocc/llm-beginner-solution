"""任务四 M3：检索器（BGE 召回 + 可选 reranker 精排）。

自检契约：
- class Retriever；构造时加载索引（不存在则现场建）
- retrieve(query: str, k: int) -> List[dict]（每项含 text / score / source）
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from .indexer import load_index

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："   # BGE 官方：query 侧必须加


class _Embedder:
    """BGE 编码器（transformers 手动池化，见 embed.py；ST 在本机会段错误）。"""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModel.from_pretrained(str(self.model_dir)).cuda().eval()
            self._torch = torch

    def encode(self, texts):
        self._ensure()
        from .embed import encode_hf
        return encode_hf(self._model, self._tok, texts)


class Retriever:
    def __init__(self, reranker=None):
        self.index, self.chunks = load_index()
        self.embedder = _Embedder(Path(__file__).resolve().parents[1] / "models" / "bge-small-zh-v1.5")
        if reranker is None:
            rr_dir = Path(__file__).resolve().parents[1] / "models" / "bge-reranker-base"
            if (rr_dir / "model.safetensors").exists():
                from .reranker import Reranker
                reranker = Reranker()
        self.reranker = reranker

    def _embed_query(self, query):
        return self.embedder.encode([QUERY_PREFIX + query])[0]

    def retrieve(self, query, k=10):
        q = self._embed_query(query)
        idx = getattr(self, "_last_top", None) or k * 2
        top = int(idx)
        scores, ids = self.index.search(q[None, :], top)
        docs = [{
            "text": self.chunks[i]["text"],
            "score": float(s),
            "source": self.chunks[i]["source"],
        } for s, i in zip(scores[0], ids[0]) if i >= 0]
        if self.reranker is not None:
            docs = self.reranker.rerank(query, docs, k)
        else:
            docs = docs[:k]
        self._last_top = top
        return docs
