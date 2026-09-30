import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.agent import ReActAgent

agent = ReActAgent(max_steps=8)
tasks = json.loads(open("data/tasks.json", encoding="utf-8").read())
for i in [0, 7, 1]:
    t = tasks[i]
    trace = agent.run(t["task"])
    print(f"=== 题目 {i+1} === | 答案: {trace['final_answer'][:150]}")
    for s in trace["steps"]:
        print("  ", s.get("step"), s.get("kind"), s.get("tool"),
              str(s.get("observation"))[:60])
