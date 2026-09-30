"""汇总打印六个任务的官方自检结果（读取各任务 eval/result.json，不重新跑实验）。

用法（仓库根目录）：
    python scripts/show_results.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

TASKS = [
    (1, "task-1-transformer", "Transformer 分类器"),
    (2, "task-2-mini-gpt", "从零实现 mini-GPT"),
    (3, "task-3-sft-dpo", "SFT + DPO 对齐"),
    (4, "task-4-rag", "RAG 文档问答"),
    (5, "task-5-tool-agent", "工具调用 Agent（ReAct）"),
    (6, "task-6-coding-agent", "Mini Coding Agent"),
]

MARK = {True: "通过", False: "未通过", None: "跳过"}


def main():
    print("=" * 96)
    print("LLM-Beginner 任务一~六 · 官方自检结果汇总（eval/result.json）")
    print("=" * 96)
    total = passed = skipped = 0
    for idx, name, title in TASKS:
        p = ROOT / name / "eval" / "result.json"
        print(f"\n【任务{'一二三四五六'[idx - 1]}】{title}   （{name}）")
        if not p.exists():
            print("    (缺少 eval/result.json，请先运行该任务的 eval/run.py)")
            continue
        for item in json.loads(p.read_text(encoding="utf-8")):
            test = item.get("test", "?")
            ok = item.get("pass")
            total += 1
            passed += int(ok is True)
            skipped += int(ok is None)
            extra = {k: v for k, v in item.items()
                     if k not in ("test", "pass") and not isinstance(v, (list, dict))}
            brief = json.dumps(extra, ensure_ascii=False)
            brief = "" if brief == "{}" else f"  {brief[:88]}"
            print(f"    [{MARK[ok]}] {test}{brief}")
    print("\n" + "=" * 96)
    print(f"合计 {total} 项自检：通过 {passed} 项，跳过 {skipped} 项（可选加分项），未通过 {total - passed - skipped} 项")
    print("=" * 96)


if __name__ == "__main__":
    main()
