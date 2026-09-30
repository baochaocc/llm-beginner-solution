"""跑全部 10 题并打印每题的成功/失败（与 eval 同口径的关键字判断）。"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import ReActAgent

ROOT = Path(__file__).resolve().parents[1]


def normalize_answer(text):
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def answer_matches(answer, expected_keywords):
    norm = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(k) in norm for k in expected):
                return False
        elif normalize_answer(expected) not in norm:
            return False
    return True


def main():
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    agent = ReActAgent(max_steps=12)
    ok = 0
    for t in tasks:
        t0 = time.time()
        try:
            trace = agent.run(t["task"])
            final = trace.get("final_answer", "")
        except Exception as e:  # noqa: BLE001
            final, trace = f"[异常] {e}", {"steps": []}
        good = answer_matches(final, t.get("expected_answer_contains", []))
        ok += int(good)
        tools = [s.get("tool") for s in trace.get("steps", []) if s.get("kind") == "action"]
        print(f"[{'✅' if good else '❌'}] {t['id']}. {t['task'][:40]}...")
        print(f"     答案: {str(final)[:100]}")
        print(f"     工具: {tools} | 步数: {len(trace.get('steps', []))} | {time.time()-t0:.0f}s")
    print(f"\n成功率: {ok}/{len(tasks)} = {ok/len(tasks):.1%}")


if __name__ == "__main__":
    main()
