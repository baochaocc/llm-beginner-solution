"""调试：跑 task 6 和 task 9，打印每一步的模型回复与 Observation。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import ReActAgent

TASKS = {
    6: "用 Python 写一个函数判断回文，并测试 'level' 和 'world' 两个字符串，输出结果分别是 True 还是 False。",
    9: "查维基百科 Transformer (机器学习模型) 的发表年份，并算到 2026 年过了多少年。",
}

agent = ReActAgent(max_steps=12)
for tid, task in TASKS.items():
    print("=" * 70)
    print(f"T{tid}: {task}")
    trace = agent.run(task)
    for s in trace.get("steps", []):
        print("--" * 30)
        print(json.dumps(s, ensure_ascii=False, indent=2)[:2500])
    print("FINAL:", trace.get("final_answer"))
