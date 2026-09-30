"""一键复现：依次运行六个任务的官方自检，保存原始日志并渲染成终端风格截图。

用法（在仓库根目录）：
    python scripts/run_all_evals.py            # 全部六个任务
    python scripts/run_all_evals.py 1 5        # 只跑指定任务

产物：
    docs/screenshots/raw/task-N-*.txt          # 原始控制台输出（stdout+stderr）
    docs/screenshots/env.png                   # 实验环境（Python/torch/GPU/关键依赖）
    docs/screenshots/task-N-*.png              # 每个任务的自检截图
    docs/screenshots/summary.png               # 汇总（各任务 result.json 的判定）
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 中文 Windows 控制台默认 GBK，强制 UTF-8 并容忍不可解码字符，避免打印时崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
RAW = OUT / "raw"

TASKS = [
    (1, "task-1-transformer", "Transformer 分类器"),
    (2, "task-2-mini-gpt", "从零实现 mini-GPT"),
    (3, "task-3-sft-dpo", "SFT + DPO 对齐"),
    (4, "task-4-rag", "RAG 文档问答"),
    (5, "task-5-tool-agent", "工具调用 Agent（ReAct）"),
    (6, "task-6-coding-agent", "Mini Coding Agent"),
]

FONT_PATH = r"C:\Windows\Fonts\simsun.ttc"      # 宋体：ASCII 半宽 + 中文全宽，终端对齐最佳
FONT_SIZE = 15
LINE_H = 21
PAD = 14
TITLE_H = 34
BG = (28, 28, 30)
FG = (219, 219, 219)
TITLE_BG = (58, 58, 62)
TITLE_FG = (240, 240, 240)
PROMPT_FG = (126, 214, 130)
WRAP_CHARS = 116


def _font(size=FONT_SIZE):
    return ImageFont.truetype(FONT_PATH, size)


def _disp_width(text):
    """显示宽度：中文/全角算 2，其余算 1。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in text)


def _wrap(text, width=WRAP_CHARS):
    """按显示宽度折行，避免长行被裁掉。"""
    lines, cur = [], ""
    for ch in text:
        if _disp_width(cur + ch) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    lines.append(cur)
    return lines


def _is_noise(line: str) -> bool:
    """模型加载进度条 / 依赖下载进度这类噪声行：截图里省掉（原始日志仍完整保留）。"""
    if not line.strip():
        return False
    if "Loading weights" in line or "Materializing param" in line:
        return True
    if "it/s]" in line or "s/it]" in line or "Downloading" in line or "Fetching" in line:
        return True
    if "█" in line or "\x1b" in line:
        return True
    head = line.lstrip()[:6]
    return head[:3].isdigit() and "%" in line[:12]


def strip_noise(lines):
    kept = [l for l in lines if not _is_noise(l)]
    dropped = len(lines) - len(kept)
    if dropped:
        kept.append(f"...（已省略 {dropped} 行模型加载/下载进度条噪声，完整原始输出见 docs/screenshots/raw/）")
    return kept


def render(title, body_lines, out_path, max_lines=70):
    """把若干行文本渲染成终端风格 PNG。"""
    lines = ["PS D:\\llm-beginner-master> " + title, ""]
    for raw_line in strip_noise(body_lines):
        raw_line = raw_line.replace("\t", "    ").rstrip("\n")
        lines.extend(_wrap(raw_line) if raw_line else [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["... (完整输出见 docs/screenshots/raw/ 下同名 txt)"]

    width = PAD * 2 + int(max(_disp_width(l) for l in lines) * FONT_SIZE * 0.52) + 8
    width = max(width, 900)
    height = TITLE_H + PAD * 2 + LINE_H * len(lines)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, TITLE_H], fill=TITLE_BG)
    d.text((12, 8), "命令提示符 — 任务自检", font=_font(15), fill=TITLE_FG)
    f = _font()
    y = TITLE_H + PAD
    for i, line in enumerate(lines):
        color = PROMPT_FG if (i == 0 and line.startswith("PS ")) else FG
        d.text((PAD, y), line, font=f, fill=color)
        y += LINE_H
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def run_cmd(cmd, cwd=ROOT, timeout=3600):
    try:
        r = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or ""), r.returncode
    except subprocess.TimeoutExpired:
        return f"[超时] {' '.join(cmd)}", -1


def env_lines():
    code = (
        "import platform, torch, transformers, sys;"
        "print('Python  :', sys.version.split()[0]);"
        "print('OS      :', platform.platform());"
        "print('torch   :', torch.__version__, '| CUDA', torch.version.cuda);"
        "print('GPU     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only');"
        "print('显存    :', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB');"
        "print('transformers:', transformers.__version__)"
    )
    out, _ = run_cmd([sys.executable, "-c", code])
    extra, _ = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv"], timeout=60)
    extra = "".join(c for c in extra if c.isprintable() or c == "\n")   # 去掉解码残渣
    pkgs, _ = run_cmd([sys.executable, "-m", "pip", "list"], timeout=300)
    want = ("datasets", "faiss-cpu", "sentence-transformers", "pypdf", "modelscope",
            "mcp", "pytest", "numpy", "pandas", "matplotlib")
    picked = [l for l in pkgs.splitlines()
              if l.split(" ")[0].lower() in want]
    return (out.strip().splitlines() + [""] + extra.strip().splitlines()
            + ["", "关键依赖："] + picked)


def summary_lines():
    lines = [f"{'任务':<26}{'自检项':<34}{'结果':<10}关键指标"]
    for _, name, _ in TASKS:
        p = ROOT / name / "eval" / "result.json"
        if not p.exists():
            lines.append(f"{name:<26}{'(缺 result.json)':<34}{'—':<10}")
            continue
        for item in json.loads(p.read_text(encoding="utf-8")):
            name_ = item.get("test", "?")
            ok = item.get("pass")
            mark = {True: "通过", False: "未通过", None: "跳过"}[ok]
            extra = {k: v for k, v in item.items()
                     if k not in ("test", "pass") and not isinstance(v, (list, dict))}
            lines.append(f"{name:<26}{name_:<34}{mark:<10}{json.dumps(extra, ensure_ascii=False)[:70]}")
    return lines


def main():
    argv = [a for a in sys.argv[1:] if a != "--render-only"]
    render_only = "--render-only" in sys.argv
    want = {int(a) for a in argv} or {t[0] for t in TASKS}
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    if render_only:
        # 只根据已保存的原始日志重新渲染截图（不重跑自检）
        for idx, name, _ in TASKS:
            log = RAW / f"task-{idx}.txt"
            if log.exists():
                render(f"python {name}/eval/run.py", log.read_text(encoding="utf-8").splitlines(),
                       OUT / f"task-{idx}.png")
                print(f"re-rendered task-{idx}.png")
        for f, title, out in ((RAW / "env.txt", "环境检查", OUT / "env.png"),
                              (RAW / "summary.txt",
                               "python scripts/run_all_evals.py  （各任务 eval/result.json 汇总）",
                               OUT / "summary.png")):
            if f.exists():
                render(title, f.read_text(encoding="utf-8").splitlines(), out)
                print(f"re-rendered {out.name}")
        return

    print("== 环境信息 ==")
    e_lines = env_lines()
    (RAW / "env.txt").write_text("\n".join(e_lines), encoding="utf-8")
    render("环境检查", e_lines, OUT / "env.png")
    print("\n".join(e_lines[:8]))

    for idx, name, title in TASKS:
        if idx not in want:
            continue
        print(f"\n== 任务 {idx}：{title} ==", flush=True)
        cmd = [sys.executable, f"{name}/eval/run.py"]
        body, code = run_cmd(cmd, timeout=3600)
        text = f"$ {' '.join(cmd)}\n{body}"
        (RAW / f"task-{idx}.txt").write_text(text, encoding="utf-8")
        render(f"python {name}/eval/run.py", text.splitlines(),
               OUT / f"task-{idx}.png")
        print(body.strip()[-1200:] or f"(exit {code})", flush=True)

    s_lines = summary_lines()
    (RAW / "summary.txt").write_text("\n".join(s_lines), encoding="utf-8")
    render("python scripts/run_all_evals.py  （各任务 eval/result.json 汇总）",
           s_lines, OUT / "summary.png")
    print("\n== 汇总 ==")
    print("\n".join(s_lines))
    print(f"\n截图与日志已写入：{OUT}")


if __name__ == "__main__":
    main()
