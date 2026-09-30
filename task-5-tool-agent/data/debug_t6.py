"""调试：跑 task 6，打印每一步模型回复的 repr 与 parse_reply 结果。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import ReActAgent, parse_reply

TASK = "用 Python 写一个函数判断回文，并测试 'level' 和 'world' 两个字符串，输出结果分别是 True 还是 False。"

agent = ReActAgent(max_steps=12)
trace = agent.run(TASK)
for s in trace.get("steps", []):
    resp = s.get("response", "")
    print("=" * 60)
    print("KIND:", s.get("kind"))
    print("REPR:", repr(resp))
    print("PARSE:", parse_reply(resp))
    if "observation" in s:
        print("OBS:", repr(s["observation"])[:800])
print("FINAL:", repr(trace.get("final_answer")))
