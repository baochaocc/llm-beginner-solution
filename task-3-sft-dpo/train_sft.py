"""任务三 M3：SFT 训练（LoRA + Qwen chat 模板 + loss masking）。

用法（在 task-3-sft-dpo 目录下）：
    python train_sft.py [--samples 8000] [--lr 2e-4] [--epochs 1] [--batch 16]

前置：data/models/Qwen2.5-0.5B 存在、data/moss-sft/moss-003-sft-no-tools.jsonl.zip 存在。
产出：ckpt/sft/（adapter.bin + adapter_config.json）
"""
import argparse
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 防 Windows GBK 控制台崩

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import format_and_mask
from src.lora import inject_lora, save_lora

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"
CKPT = ROOT / "ckpt"

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
R, ALPHA = 16, 32
MAX_LEN = 384


def load_instruction_data(source_dir, num_samples, seed=42):
    """从目录里扫 jsonl/json/parquet，抽出规整后的 messages 列表。

    兼容格式：
    - {"messages": [{"role":..., "content":...}, ...]} / {"conversations": [...]}
    - 单轮：{"instruction"/"prompt"/"input": ..., "output"/"response": ...}
    - 多轮单字符串：{"prompt":..., "response":...}（按 \n 分段做多轮）
    """
    data_dir = ROOT / source_dir
    files = (sorted(data_dir.rglob("*.jsonl"))
             + sorted(data_dir.rglob("*.json"))
             + sorted(data_dir.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 jsonl/json/parquet 文件")

    def extract(obj):
        msgs = obj.get("messages") or obj.get("conversations")
        if not msgs and isinstance(obj.get("conversation"), list):
            # MOSS 格式：{"conversation": [{"human": ..., "assistant": ...}, ...]}
            conv = []
            for turn in obj["conversation"]:
                if not isinstance(turn, dict):
                    continue
                h = turn.get("human")
                a = turn.get("assistant")
                if h:
                    conv.append({"role": "user", "content": str(h).strip()})
                if a:
                    conv.append({"role": "assistant", "content": str(a).strip()})
            msgs = conv if len(conv) >= 2 else None
        if msgs:
            cleaned = []
            for m in msgs:
                role = str(m.get("role", "")).lower()
                if role not in ("system", "user", "assistant"):
                    continue
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                cleaned.append({"role": role, "content": content})
            return cleaned if len(cleaned) >= 2 else None
        # 单轮指令
        instr = obj.get("instruction") or obj.get("prompt") or obj.get("input")
        out = obj.get("output") or obj.get("response") or obj.get("answer")
        if instr and out:
            return [{"role": "user", "content": str(instr).strip()},
                    {"role": "assistant", "content": str(out).strip()}]
        return None

    items = []
    for f in files:
        try:
            if f.suffix == ".parquet":
                import pandas as pd
                rows = pd.read_parquet(f).to_dict("records")
            elif f.suffix == ".json":
                obj = json.loads(f.read_text(encoding="utf-8"))
                rows = obj if isinstance(obj, list) else [obj]
            else:
                rows = [json.loads(line) for line in
                        f.read_text(encoding="utf-8").splitlines() if line.strip()]
            for obj in rows:
                if not isinstance(obj, dict):
                    continue
                msgs = extract(obj)
                if msgs:
                    items.append(msgs)
        except Exception as e:
            print(f"[跳过] {f.name}: {type(e).__name__} {str(e)[:60]}")
    random.Random(seed).shuffle(items)
    print(f"SFT 数据共 {len(items)} 条，取前 {min(num_samples, len(items))} 条（{source_dir}）")
    return items[:num_samples]


def keep_recent_turns(messages, max_turns=5):
    """长对话只留 system + 最近 max_turns 轮（截断 token 时保留对话尾部）。"""
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return (system + [rest[-1]] + rest[-max_turns:]) if rest else system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--data_dir", type=str, default="data/moss-sft-10w",
                    help="SFT 数据目录（jsonl/json/parquet）")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    pad_id = tok.eos_token_id
    print(f"pad_id = {pad_id} (eos) | vocab {tok.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16).to(device)
    inject_lora(model, ALL_TARGETS, R, ALPHA)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA 可训练参数 {trainable / 1e6:.2f}M / 总计 {total / 1e6:.2f}M"
          f"（{100 * trainable / total:.2f}%）target={ALL_TARGETS} r={R} alpha={ALPHA}")

    # ---- 数据：MOSS -> (input_ids, labels) ----
    t0 = time.time()
    items = load_instruction_data(args.data_dir, args.samples, args.seed)
    samples = []
    for msgs in items:
        msgs = keep_recent_turns(msgs)
        try:
            ids, labels = format_and_mask(msgs, max_len=args.max_len)
        except Exception as e:
            continue
        if len(ids) >= 4 and any(l != -100 for l in labels):
            samples.append((ids, labels))
    print(f"有效样本 {len(samples)} 条（预处理 {time.time() - t0:.0f}s）")
    assert samples, "没有有效样本！检查 MOSS 数据格式"

    # ---- 训练 ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(samples) / args.batch)
    warmup = min(30, total_steps // 10)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, log_loss = 0, 0.0
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(samples)
        for i in range(0, len(samples), args.batch):
            batch = samples[i:i + args.batch]
            L = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), L), -100, dtype=torch.long)
            for j, (ids_j, labels_j) in enumerate(batch):
                ids[j, :len(ids_j)] = torch.tensor(ids_j)
                labels[j, :len(labels_j)] = torch.tensor(labels_j)
            ids, labels = ids.to(device), labels.to(device)

            model.train()
            opt.zero_grad()
            logits = model(ids).logits          # (B, T, V) bf16；转 fp32 会 3.7GB+ OOM
            # next-token：logits[t] 预测 ids[t+1]，labels 必须错位 1
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            log_loss += loss.item()
            if step % 25 == 0 or step == total_steps:
                print(f"epoch {epoch}/{args.epochs} | step {step}/{total_steps} "
                      f"| loss {log_loss / 25:.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")
                log_loss = 0.0

    # ---- 保存 ----
    save_lora(model, CKPT / "sft", R, ALPHA, ALL_TARGETS)
    print(f"已保存 ckpt/sft/（{sum(p.numel() for p in torch.load(CKPT / 'sft' / 'adapter.bin', map_location='cpu').values()) / 1e6:.2f}M 参数）")


if __name__ == "__main__":
    main()
