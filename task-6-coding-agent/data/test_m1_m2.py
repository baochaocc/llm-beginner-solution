"""M1/M2 快速自测：MCP 工具列表 + Skill loader。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcp_server import list_tools, dispatch
from src.skill_loader import SkillLoader

tools = list_tools()
names = [t["name"] for t in tools]
print("tools:", names, "| count:", len(tools), "| all have name:", all(t.get("name") for t in tools))
print("schemas ok:", all({"name", "description", "input_schema"} <= set(t) for t in tools))

# 路径安全 + 工具功能
repo = Path(__file__).resolve().parents[1] / "data" / "toy-repo"
print("read_file:", dispatch("read_file", {"path": "calculator.py"}, repo)[:80].replace("\n", " | "))
print("run_tests:", dispatch("run_tests", {}, repo).replace("\n", " | ")[:120])
print("git_diff:", dispatch("git_diff", {}, repo).replace("\n", " | ")[:120])
for bad in ("../README.md", str(repo / "calculator.py"), "..\\..\\README.md"):
    out = dispatch("read_file", {"path": bad}, repo)
    print(f"安全拦截 {bad!r}: {'OK' if '工具错误' in out else '失败!'} -> {out[:60]}")

loader = SkillLoader(str(Path(__file__).resolve().parents[1] / "src" / "skills"))
skills = loader.list_skills()
print("skills:", [(s["name"], bool(s["description"])) for s in skills])
print("load test-runner:", loader.load("test-runner")[:60].replace("\n", " | "))
