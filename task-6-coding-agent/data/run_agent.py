"""M3 快速试跑：重置 buggy 快照 -> CodingAgent.run -> 打印 Trace 摘要。"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import CodingAgent

ROOT = Path(__file__).resolve().parents[1]
repo = ROOT / "data" / "toy-repo"
# 恢复干净 toy repo：buggy 快照重置 + 清掉模型可能产生的杂物文件
shutil.copy(repo / "calculator.py.orig", repo / "calculator.py")
KEEP = {"calculator.py", "calculator.py.orig", "test_calculator.py", "ISSUE.md"}
for p in repo.iterdir():
    if p.name not in KEEP:
        shutil.rmtree(p) if p.is_dir() else p.unlink()

issue = (repo / "ISSUE.md").read_text(encoding="utf-8")
agent = CodingAgent(max_steps=8)
t0 = time.time()
trace = agent.run(repo_path=str(repo), issue=issue)
print(f"\n{'=' * 60}\n耗时 {time.time() - t0:.0f}s")
print("tests_passed:", trace["tests_passed"])
print("done_reason:", trace.get("done_reason"))
print("final_answer:", str(trace["final_answer"])[:300])
print("n_steps:", trace["n_steps"], "| patch 长度:", len(trace["patch"]))
print("\n--- steps ---")
for s in trace["steps"]:
    if s.get("kind") == "action":
        print(f"[{s['step']}] {s['tool']} {json.dumps(s.get('args', {}), ensure_ascii=False)[:80]}")
        print(f"     obs: {str(s.get('observation'))[:150].replace(chr(10), ' | ')}")
    elif s.get("kind") == "final":
        print(f"[{s['step']}] FINAL: {str(s.get('final'))[:120]}")
    else:
        print(f"[{s['step']}] {s.get('kind')}: {str(s.get('response') or s)[:100]}")
print("\n--- subagents ---")
for sa in trace.get("subagent_reports", []):
    print(json.dumps(sa, ensure_ascii=False)[:200])
