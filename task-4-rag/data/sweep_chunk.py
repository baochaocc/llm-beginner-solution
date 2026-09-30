"""chunk_size 扫描：每种尺寸重建索引并跑 gold 召回（找最优分块）。"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import numpy as np
import faiss

from src.indexer import extract_pdf_pages, INDEX_DIR
from src.chunker import chunk_documents
from src.embed import encode_hf

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def normalize_text(t):
    return re.sub(r"\s+", "", str(t))


def build(size, overlap):
    pages = extract_pdf_pages(ROOT / "data" / "kb.pdf")
    chunks = chunk_documents(pages, size, overlap)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "bge-small-zh-v1.5"))
    m = AutoModel.from_pretrained(str(ROOT / "models" / "bge-small-zh-v1.5")).cuda().eval()
    emb = encode_hf(m, tok, [c[0] for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    return index, chunks


def recall(index, chunks, embedder, gold):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks = []
    for item in gold:
        anchors = [normalize_text(a) for a in item["gold_anchors"]]
        q = embedder.encode(["为这个句子生成表示以用于检索相关文章：" + item["question"]])[0]
        scores, ids = index.search(q[None, :], 10)
        rank_matched = None
        for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
            if i < 0:
                continue
            text = normalize_text(chunks[i][0])
            if any(a and a in text for a in anchors):
                rank_matched = rank
                break
        if rank_matched:
            for k in hits:
                if rank_matched <= k:
                    hits[k] += 1
            ranks.append(1 / rank_matched)
        else:
            ranks.append(0)
    return {f"r@{k}": round(v / len(gold), 3) for k, v in hits.items()} | {
        "mrr": round(sum(ranks) / len(gold), 3)}


def main():
    gold = [json.loads(l) for l in
            (ROOT / "data" / "gold_qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "bge-small-zh-v1.5"))
    m = AutoModel.from_pretrained(str(ROOT / "models" / "bge-small-zh-v1.5")).cuda().eval()

    class E:
        def encode(self, texts):
            return encode_hf(m, tok, texts)

    # 再试两档：768/128 与「按页为单元」（页长 >1200 才细分）
    from src.chunker import clean_text
    for size, overlap in [(768, 128)]:
        index, chunks = build(size, overlap)
        r = recall(index, chunks, E(), gold)
        print(f"size={size:5d} overlap={overlap:4d} chunks={len(chunks):5d}  {r}")

    # 按页为 chunk 单元：每页文本 <=1200 字则整页一块，否则按 768/128 细分
    from src.chunker import chunk_text
    pages = extract_pdf_pages(ROOT / "data" / "kb.pdf")
    page_chunks = []
    for text, src in pages:
        t = clean_text(text)
        if len(t) <= 1200 and t.strip():
            page_chunks.append((t, src))
        else:
            for p in chunk_text(t, 768, 128):
                page_chunks.append((p, src))
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "bge-small-zh-v1.5"))
    m = AutoModel.from_pretrained(str(ROOT / "models" / "bge-small-zh-v1.5")).cuda().eval()
    emb = encode_hf(m, tok, [c[0] for c in page_chunks])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    r = recall(index, page_chunks, E(), gold)
    print(f"page-aware            chunks={len(page_chunks):5d}  {r}")


def split_piece(text, size, overlap):
    from src.chunker import chunk_text
    return chunk_text(text, size, overlap)


if __name__ == "__main__":
    main()
