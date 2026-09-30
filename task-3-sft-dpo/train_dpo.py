"""任务三 M4：DPO 训练（在 SFT 之上继续偏好对齐）。

用法（在 task-3-sft-dpo 目录下）：
    python train_dpo.py [--pairs 4000] [--lr 1e-4] [--epochs 1] [--batch 8] [--beta 0.1]

前置：ckpt/sft/ 存在；data/dpo/dpo_{zh,en}.json 存在。
产出：ckpt/dpo/（adapter.bin + adapter_config.json）+ results/dpo_curve.json

DPO 损失：-log σ(β * [ (log πθ(chosen) - log πθ(rejected))
                      - (log πref(chosen) - log πref(rejected)) ])
- policy = base + SFT LoRA 权重（继续训练，只更新 LoRA 参数）
- ref    = base + SFT LoRA 权重（冻结，只前向；用 SFT 模型作参考是 DPO 标准做法）
- logp 取 response 部分 token 的均值（比 sum 更稳定）
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import format_messages
from src.lora import load_lora, save_lora

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"
CKPT = ROOT / "ckpt"

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]


def load_dpo_pairs(data_dir, num_pairs, seed=42):
    """从目录读偏好对（instruction/chosen/rejected），兼容 json/jsonl/parquet。"""
    data_dir = ROOT / data_dir
    files = (sorted(data_dir.rglob("*.jsonl")) + sorted(data_dir.rglob("*.json"))
             + sorted(data_dir.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 json/jsonl/parquet")

    rows = []
    for p in files:
        try:
            if p.suffix == ".parquet":
                rows += pd.read_parquet(p).to_dict("records")
            elif p.suffix == ".jsonl":
                rows += [json.loads(line) for line in
                         p.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                rows += obj if isinstance(obj, list) else [obj]
        except Exception as e:
            print(f"[跳过] {p.name}: {type(e).__name__} {str(e)[:60]}")

    norm = [{str(k).lower(): v for k, v in row.items()} for row in rows if isinstance(row, dict)]

    def pick(obj, *names):
        for n in names:
            if n in obj and obj[n]:
                return obj[n]
        return None

    pairs = []
    for obj in norm:
        instr = pick(obj, "instruction", "prompt", "question", "input")
        ch = pick(obj, "chosen", "chosen_response", "response0", "positive", "output", "answer")
        rej = pick(obj, "rejected", "rejected_response", "response1", "negative")
        if instr is None:
            continue
        # 有些数据集字段是 instruction/正负对，而非 chosen/rejected
        if ch is not None and rej is not None:
            instr, ch, rej = str(instr).strip(), str(ch).strip(), str(rej).strip()
            if instr and ch and rej and len(ch) > 4 and len(rej) > 4:
                pairs.append((instr, ch, rej))
    random.Random(seed).shuffle(pairs)
    print(f"DPO 数据共 {len(pairs)} 对，取前 {min(num_pairs, len(pairs))} 对（{data_dir}）")
    return pairs[:num_pairs]


def tokenize_pair(tok, pair, max_len):
    """一对 -> (ids_ch, mask_ch, ids_rej, mask_rej)；mask=1 表示 response 部分 token。"""
    instr, ch, rej = pair
    marker_ids = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    out = []
    for resp in (ch, rej):
        text = format_messages([
            {"role": "user", "content": instr},
            {"role": "assistant", "content": resp},
        ])
        ids = tok(text, add_special_tokens=False, truncation=True,
                  max_length=max_len)["input_ids"]
        # response 起点 = 最后一个 <|im_start|>assistant\n 之后
        start = -1
        for k in range(len(ids) - len(marker_ids), -1, -1):
            if ids[k:k + len(marker_ids)] == marker_ids:
                start = k + len(marker_ids)
                break
        if start < 0:
            start = 0
        mask = [0] * len(ids)
        for t in range(start, len(ids)):
            mask[t] = 1
        out.append((ids, mask))
    return out


def gather_logp(logits, ids, mask):
    """logits (B,T,V)、ids (B,T)、mask (B,T) -> 每个样本 response 平均 logp (B,)。

    next-token 口径：logits[t] 预测 ids[t+1]，所以按错位取对数概率：
    log_probs[:, :-1] 对 ids[:, 1:] 聚集，mask 也取 [:, 1:]。
    全程 bf16（fp32 log_softmax 在 15 万词表上每步多分配 ~8GB，会拖慢到 30s/步）。
    """
    log_probs = F.log_softmax(logits, dim=-1)                          # (B, T, V) bf16
    tok_lp = log_probs[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()  # (B, T-1) fp32
    mask = mask[:, 1:]
    tok_lp = tok_lp.masked_fill(mask == 0, 0.0)
    return tok_lp.sum(-1) / mask.sum(-1).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--data_dir", type=str, default="data/dpo-humanlike",
                    help="DPO 偏好数据目录（json/jsonl/parquet）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    pad_id = tok.eos_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16).to(device)
    load_lora(policy, CKPT / "sft", target_modules=ALL_TARGETS)
    ref = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16).to(device)
    load_lora(ref, CKPT / "sft", target_modules=ALL_TARGETS)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    print(f"policy 可训练参数："
          f"{sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.2f}M")

    pairs = load_dpo_pairs(args.data_dir, args.pairs, args.seed)
    t0 = time.time()
    pre = []
    for p in pairs:
        (ids_ch, mask_ch), (ids_rej, mask_rej) = tokenize_pair(tok, p, args.max_len)
        pre.append((ids_ch, mask_ch, ids_rej, mask_rej))
    print(f"预 tokenize {len(pre)} 对（{time.time() - t0:.0f}s）")

    def pad_batch(seqs):
        L = max(len(p[0]) for p in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, (ids_j, mask_j) in enumerate(seqs):
            ids[j, :len(ids_j)] = torch.tensor(ids_j)
            mask[j, :len(mask_j)] = torch.tensor(mask_j)
        return ids.to(device), mask.to(device)

    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(pre) / args.batch)
    warmup = min(20, total_steps // 10)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def forward_logps(model, batch, with_grad):
        """返回 (pol_ch, pol_rej)；policy 需梯度时走正常前向，ref 走 no_grad。"""
        ids_ch, mask_ch = pad_batch([(s[0], s[1]) for s in batch])
        ids_rej, mask_rej = pad_batch([(s[2], s[3]) for s in batch])
        if with_grad:
            return (gather_logp(model(ids_ch).logits, ids_ch, mask_ch),
                    gather_logp(model(ids_rej).logits, ids_rej, mask_rej))
        with torch.no_grad():
            return (gather_logp(model(ids_ch).logits, ids_ch, mask_ch),
                    gather_logp(model(ids_rej).logits, ids_rej, mask_rej))

    hist = {"loss": [], "margin": [], "win_rate": []}
    step, log_loss = 0, 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(pre)
        for i in range(0, len(pre), args.batch):
            t_step = time.time()
            batch = pre[i:i + args.batch]

            # policy：需要梯度（只更新 LoRA）
            policy.train()
            opt.zero_grad()
            t_a = time.time()
            pol_ch, pol_rej = forward_logps(policy, batch, True)
            if step < 2:
                print(f"  [timing s{step}] policy fwd: {time.time() - t_a:.2f}s")
            # ref：冻结
            t_b = time.time()
            ref_ch, ref_rej = forward_logps(ref, batch, False)
            if step < 2:
                print(f"  [timing s{step}] ref fwd: {time.time() - t_b:.2f}s")

            t_c = time.time()
            d_pol = pol_ch - pol_rej
            d_ref = ref_ch - ref_rej
            loss = -F.logsigmoid(args.beta * (d_pol - d_ref)).mean()
            loss.backward()
            if step < 2:
                print(f"  [timing s{step}] loss+backward: {time.time() - t_c:.2f}s")
            t_d = time.time()
            nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step < 3:
                print(f"  [timing] opt step: {time.time() - t_d:.2f}s | 总 {time.time() - t_step:.2f}s")
            log_loss += loss.item()

            with torch.no_grad():
                win = (d_pol > 0).float().mean().item()
            hist["loss"].append(loss.item())
            hist["margin"].append(d_pol.mean().item())
            hist["win_rate"].append(win)

            if step % 50 == 0 or step == total_steps:
                m = history_mean(hist, 50)
                print(f"epoch {epoch}/{args.epochs} | step {step}/{total_steps} "
                      f"| loss {log_loss / 50:.4f} | margin {m['margin']:.4f} "
                      f"| win_rate {m['win_rate']:.3f} | lr {sched.get_last_lr()[0]:.2e} "
                      f"| {time.time() - t0:.0f}s")
                log_loss = 0.0

    save_lora(policy, CKPT / "dpo", **read_lora_cfg(CKPT / "sft"))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dpo_curve.json").write_text(
        json.dumps(hist, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 ckpt/dpo/ 和 results/dpo_curve.json")


def history_mean(hist, n):
    tail = {k: v[-n:] for k, v in hist.items()}
    return {k: sum(v) / len(v) for k, v in tail.items()}


def read_lora_cfg(path):
    import json as _json
    cfg = _json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    return {"r": cfg["r"], "alpha": cfg["alpha"], "target_modules": cfg["target_modules"]}


if __name__ == "__main__":
    main()
