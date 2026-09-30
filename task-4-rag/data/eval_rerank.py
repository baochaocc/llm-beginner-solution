"""S2：加/不加 reranker 的检索质量对比（gold QA 上 R@1/3/5/10 + MRR）。"""
import json
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.retriever import Retriever
from src.reranker import Reranker


def normalize_text(t):
    return re.sub(r"\s+", "", str(t))


def evaluate(retriever, gold):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks = []
    for item in gold:
        anchors = [normalize_text(a) for a in item["gold_anchors"]]
        results = retriever.retrieve(item["question"], k=10)
        hit_rank = None
        for rank, r in enumerate(results, 1):
            text = normalize_text(r.get("text", ""))
            if any(a and a in text for a in anchors):
                hit_rank = rank
                break
        if hit_rank:
            for k in hits:
                if hit_rank <= k:
                    hits[k] += 1
            ranks.append(1 / hit_rank)
        else:
            ranks.append(0)
    return {f"recall_at_{k}": round(v / len(gold), 3) for k, v in hits.items()} | {
        "mrr": round(sum(ranks) / len(gold), 3)}


def main():
    gold = [json.loads(l) for l in
            (__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "gold_qa.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    t0 = time.time()
    plain = Retriever()
    r_plain = evaluate(plain, gold)
    print(f"无 reranker（召回 10 直出）: {r_plain}  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    with_rerank = Retriever(reranker=Reranker())
    r_rerank = evaluate(with_rerank, gold)
    print(f"有 reranker（召回 20 -> 精排 -> 10）: {r_rerank}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
