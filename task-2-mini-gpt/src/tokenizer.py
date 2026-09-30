"""任务二 M1：手写简化版 BPE tokenizer（byte-level / char-level 双模式）。

设计（与 GPT-2 思路一致，但完全手写、不依赖 tiktoken/sentencepiece）：

**byte-level 模式**（GPT-2 风格）：
1. 文本 -> UTF-8 字节流，256 个字节是初始 token；
2. 迭代合并最高频相邻 token 对，直到词表达到目标大小。

**char-level 模式**（中文更优：字符是语义单元，预测目标更干净）：
1. 字符级初始词表 = 256 个字节（回退用）+ 训练语料里出现的所有字符；
2. 对字符序列做同样的相邻对合并（这就是"字符级 BPE"）；
3. 编码时遇到训练语料里没出现过的字符（如 dev 生僻字）：
   直接回退成它的 UTF-8 字节（id 0-255 恒在词表里），保证 **encode->decode 永远精确还原**。

- decode 统一实现：token id -> 字节串，先拼接成完整字节流再整体 UTF-8 解码
  （不能逐个 token 解码：merged token 可能横跨一个汉字的多个字节）。
"""
import base64
import json
from collections import Counter
from pathlib import Path


def _merge_pair(ids, a, b, new_id):
    """把 ids 中所有相邻的 (a, b) 替换成 new_id。"""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, token_bytes, merges, chars=None):
        """token_bytes: list[bytes]，下标即 token id（0-255 恒为单字节）；
        merges: list[(a, b)]，按训练时合并顺序（rank 越小越优先）；
        chars: char-level 模式的字符列表（id = 256 + index），byte 模式为 None。"""
        self.token_bytes = token_bytes
        self.merges = merges
        self.chars = chars
        self.mode = "char" if chars is not None else "byte"
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.pair_to_new_id = {pair: 256 + len(chars or []) + i for i, pair in enumerate(merges)}
        self.char_to_id = {c: 256 + i for i, c in enumerate(chars or [])}
        self.vocab_size = len(token_bytes)

    # ---------- 训练 ----------
    @classmethod
    def train(cls, text: str, vocab_size: int = 512, level: str = "char"):
        """level="byte"：字节级（GPT-2 式）；level="char"：字符级 + 字节回退。"""
        if level == "byte":
            ids = list(text.encode("utf-8"))
            token_bytes = [bytes([i]) for i in range(256)]
            chars = None
        elif level == "char":
            # 字符按出现频次排序（高频在前，id 更小、越可能被更早 merge）
            chars = sorted(set(text), key=lambda c: -text.count(c))
            token_bytes = [bytes([i]) for i in range(256)] + [c.encode("utf-8") for c in chars]
            ids = [256 + chars.index(c) for c in text]
        else:
            raise ValueError(f"未知 level: {level}")

        merges = []
        while len(token_bytes) < vocab_size:
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < 2:                       # 没有重复出现的对，没必要再合
                break
            new_id = len(token_bytes)
            token_bytes.append(token_bytes[a] + token_bytes[b])
            merges.append((a, b))
            ids = _merge_pair(ids, a, b, new_id)   # 按新词表重新编码语料

        return cls(token_bytes, merges, chars)

    # ---------- 编码 / 解码 ----------
    def encode(self, text: str):
        """text -> List[int]。贪心按 merge 优先级合并（GPT-2 风格）。"""
        if self.mode == "char":
            ids = []
            for c in text:
                cid = self.char_to_id.get(c)
                if cid is not None:
                    ids.append(cid)
                else:
                    ids.extend(c.encode("utf-8"))   # 未登录字符 -> 字节回退
        else:
            ids = list(text.encode("utf-8"))

        changed = True
        while changed:
            changed = False
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                rank = self.merge_ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i >= 0:
                new_id = self.pair_to_new_id[(ids[best_i], ids[best_i + 1])]
                ids = ids[:best_i] + [new_id] + ids[best_i + 2:]
                changed = True
        return ids

    def decode(self, ids):
        """List[int] -> str。字节串拼接后整体 UTF-8 解码（见模块 docstring）。"""
        raw = b"".join(self.token_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ---------- 持久化 ----------
    def save(self, path):
        data = {
            "mode": self.mode,
            "merges": [[a, b] for a, b in self.merges],
            # 字节串不能直接进 JSON，用 base64
            "bytes": [base64.b64encode(b).decode("ascii") for b in self.token_bytes],
        }
        if self.chars is not None:
            data["chars"] = self.chars
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        token_bytes = [base64.b64decode(s) for s in data["bytes"]]
        merges = [tuple(p) for p in data["merges"]]
        chars = data.get("chars")
        return cls(token_bytes, merges, chars)
