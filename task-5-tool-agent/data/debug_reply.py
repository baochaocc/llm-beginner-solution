import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.agent import ReActAgent

agent = ReActAgent(max_steps=3)
t = json.loads(open("data/tasks.json", encoding="utf-8").read())[0]
trace = agent.run(t["task"])
print("FINAL:", repr(trace["final_answer"])[:300])
for s in trace["steps"]:
    print("STEP", s.get("step"), s.get("kind"), s.get("tool"), s.get("action_input"))
    print("  RESPONSE:", repr(s.get("response", ""))[:400])
    print("  OBS:", repr(s.get("observation", ""))[:200])
