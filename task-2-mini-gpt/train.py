"""任务二 M4：mini-GPT 预训练脚本。

用法（在 task-2-mini-gpt 目录下）：
    python train.py                       # 默认配置
    python train.py --d_model 192 --n_layers 6 --vocab_size 512 --epochs 200

前置：python data/download.py（生成 data/train.txt / dev.txt / dataset_info.json）
产出：
- ckpt/tokenizer.json  训练好的 byte-level BPE 词表（M1）
- ckpt/best.pt         按 dev ppl 选的最优 state_dict（M4）
- ckpt/config.json     模型超参（load_for_eval 依赖）
- history.json + figures/ppl_curve.png  训练曲线
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

assert hasattr(sys.stdout, "reconfigure")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 防 Windows GBK 控制台崩

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CKPT = ROOT / "ckpt"
FIG = ROOT / "figures"

# ---- 超参默认值（最优配置：唐诗 quick-start 消融后的选择；命令行可覆盖）----
VOCAB_SIZE_DEFAULT = 352        # 256 字节 + 96 merges（小语料上词表越小 ppl 越好）
LEVEL_DEFAULT = "byte"          # byte-level；char-level 在极小语料上会崩（见 EXPERIMENT_NOTES.md）
D_MODEL_DEFAULT = 256
N_HEADS_DEFAULT = 8
N_LAYERS_DEFAULT = 6
D_FF_DEFAULT = 1024
BLOCK_SIZE_DEFAULT = 128    # 训练上下文长度；也作为 block_size 暴露给自检
BATCH_DEFAULT = 64
EPOCHS_DEFAULT = 300
LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 50
DROPOUT_DEFAULT = 0.25
EARLY_STOP_PATIENCE = 20
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_batch(ids, batch_size, block_size, device):
    """随机切 batch 个 (block+1) 长度的窗口（next-token 需要 +1）。"""
    high = len(ids) - block_size - 1
    starts = torch.randint(0, max(1, high), (batch_size,))
    xs = [torch.tensor(ids[s:s + block_size + 1], dtype=torch.long) for s in starts]
    return torch.stack(xs).to(device)


@torch.no_grad()
def dev_perplexity(model, ids, block_size, device):
    """与自检同口径：非重叠 [block, block+1] 窗口累加 NLL 求 ppl。"""
    model.eval()
    total_nll, n = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i:i + block_size + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)
        total_nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1), reduction="sum").item()
        n += len(window) - 1
    return math.exp(total_nll / max(n, 1))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=VOCAB_SIZE_DEFAULT)
    ap.add_argument("--level", choices=["char", "byte"], default=LEVEL_DEFAULT,
                    help="BPE 级别：byte（GPT-2 式，推荐）/ char（字符级+字节回退）")
    ap.add_argument("--d_model", type=int, default=D_MODEL_DEFAULT)
    ap.add_argument("--n_heads", type=int, default=N_HEADS_DEFAULT)
    ap.add_argument("--n_layers", type=int, default=N_LAYERS_DEFAULT)
    ap.add_argument("--d_ff", type=int, default=D_FF_DEFAULT)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--dropout", type=float, default=DROPOUT_DEFAULT)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--block", type=int, default=BLOCK_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_text = (DATA / "train.txt").read_text(encoding="utf-8")
    dev_text = (DATA / "dev.txt").read_text(encoding="utf-8")
    print(f"训练语料 {len(train_text)} 字符 / dev {len(dev_text)} 字符")

    # ---- M1：BPE 只在训练语料上训练（dev 不参与），并立即落盘 ----
    tok = BPETokenizer.train(train_text, args.vocab_size, level=args.level)
    CKPT.mkdir(exist_ok=True)
    tok.save(CKPT / "tokenizer.json")
    ids_tr = tok.encode(train_text)
    ids_dev = tok.encode(dev_text)
    print(f"词表 {tok.vocab_size} | train {len(ids_tr)} tokens | dev {len(ids_dev)} tokens")

    model = MiniGPT(vocab_size=tok.vocab_size, d_model=args.d_model,
                    n_heads=args.n_heads, n_layers=args.n_layers,
                    d_ff=args.d_ff, max_seq_len=args.block,
                    dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.2f}M | block_size={args.block} | batch={args.batch}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(ids_tr) / (args.batch * args.block)))
    total_steps = args.epochs * steps_per_epoch
    print(f"每 epoch {steps_per_epoch} 步（≈1 遍语料），共 {total_steps} 步")

    def lr_at(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        t = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    history = {"train_loss": [], "dev_ppl": [], "lr": []}
    best_ppl, best_state, patience = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, steps = 0.0, 0
        for _ in range(steps_per_epoch):
            x = random_batch(ids_tr, args.batch, args.block, device)
            opt.zero_grad()
            logits = model(x[:, :-1])                      # (B, block, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                x[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            steps += 1
        train_loss = tot_loss / steps

        ppl = dev_perplexity(model, ids_dev, args.block, device)
        cur_lr = sched.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["dev_ppl"].append(ppl)
        history["lr"].append(cur_lr)
        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | dev ppl {ppl:8.2f} | lr {cur_lr:.2e} | {time.time()-t0:.0f}s")

        if ppl < best_ppl:
            best_ppl = ppl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"early stop（dev ppl 连续 {EARLY_STOP_PATIENCE} epoch 未改善）")
                break

    assert best_state is not None
    torch.save(best_state, CKPT / "best.pt")
    cfg = {"model": {
        "vocab_size": tok.vocab_size, "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "d_ff": args.d_ff, "max_seq_len": args.block,
        "dropout": args.dropout,
    }}
    (CKPT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best dev ppl = {best_ppl:.2f} -> 已保存 ckpt/best.pt")

    # 训练曲线
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Train loss")
    ax[1].plot(history["dev_ppl"], label="dev ppl")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("perplexity"); ax[1].legend(); ax[1].set_title("Dev perplexity")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "ppl_curve.png", dpi=150)
    print(f"训练曲线已保存 figures/ppl_curve.png")

    # 顺手看一眼 greedy 生成
    model.eval()
    prompt = "床前明月光"
    gen = model.generate(tok.encode(prompt), max_new_tokens=30, temperature=0.0)
    print(f"\n[greedy 示例] {tok.decode(gen)}")


if __name__ == "__main__":
    train(parse_args())
