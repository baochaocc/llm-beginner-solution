"""任务六（M3 附加）：Subagent 基类 + 两个预置子代理（代码搜索 / 测试执行）。

隔离性检查点（三样缺一不可）：
1. 独立 message 列表（不共享主 agent 上下文）；
2. 独立步数上限（子代理空转不会拖死主循环）；
3. 工具子集（只给最小必要工具，减少越权行为面）。
"""
import json
import re

from .. import mcp_server as mcp


class SubAgent:
    """一个可独立运行的子代理：own context + own step cap + tool subset。"""

    def __init__(self, name: str, system_prompt: str, allowed_tools: list,
                 llm=None, max_steps: int = 4, summary_limit: int = 500):
        self.name = name
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.llm = llm
        self.max_steps = max_steps
        self.summary_limit = summary_limit

    def _extract_json(self, text):
        m = re.search(r"\{.*\}", text, re.S)     # 子代理只要一个 JSON 对象
        if not m:
            return None
        for extra in ("", "}", "}}", "}}}"):
            try:
                return json.loads(m.group(0) + extra)
            except json.JSONDecodeError:
                continue
        return None

    def run(self, task: str, repo) -> str:
        """独立 mini-loop（模型 -> 工具 -> 观察 -> 结论），返回有界摘要。"""
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}]
        summary = ""
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages, max_tokens=350)
            obj = self._extract_json(reply)
            # 工具调用 -> 执行并喂回；结论 -> 记为摘要并退出
            if obj and ("final" in obj or "answer" in obj):
                summary = str(obj.get("final") or obj.get("answer") or reply)
                break
            if obj and ("tool" in obj or "action" in obj):
                act = obj.get("action") or obj
                tool = str(act.get("tool") or "").lower()
                args = act.get("args") or act.get("arguments") or {}
                if tool in self.allowed_tools:
                    obs = mcp.dispatch(tool, args, repo)
                else:
                    obs = f"工具 {tool} 不在子代理可用范围：{self.allowed_tools}"
                messages.append({"role": "assistant", "content": reply[:2000]})
                messages.append({"role": "user", "content": f"观察：{obs[:1500]}"})
                continue
            summary = reply.strip()          # 无工具调用 -> 直接当结论
            break
        return summary[:self.summary_limit]


EXPLORE_SYSTEM = (
    "你是子代理「代码搜索」，只负责帮主代理把 repo 结构摸清楚。\n"
    "可用工具：read_file / list_dir / search_files（只读，不许改文件）。\n"
    "规则：\n"
    "1. 每步输出一个 JSON：{\"tool\": \"search_files\", \"args\": {\"pattern\": \"...\"}} 或\n"
    "   {\"final\": \"结论文本\"}；\n"
    "2. 找到关键文件/函数后立刻输出 final 结论，不要重复搜索；\n"
    "3. final 用中文，≤300 字，必须包含：涉及的文件名、可疑函数名、你认为的 bug 行。"
)

TEST_SYSTEM = (
    "你是子代理「测试执行」，负责运行测试并解释结果。\n"
    "可用工具：run_tests / read_file / search_files。\n"
    "规则：\n"
    "1. 每步输出一个 JSON：{\"tool\": \"run_tests\", \"args\": {}} 或 {\"final\": \"结论\"}；\n"
    "2. 先 run_tests，再看失败信息，可 read_file 看测试与源码；\n"
    "3. final 用中文，≤300 字：哪些用例过/挂、失败原因一句话、建议改哪里。"
)


def make_subagent(kind: str, llm, max_steps: int = 4) -> SubAgent:
    """按需创建子代理实例。"""
    if kind == "explore":
        return SubAgent("explore", EXPLORE_SYSTEM,
                        ["read_file", "list_dir", "search_files"], llm=llm,
                        max_steps=max_steps, summary_limit=500)
    if kind == "test":
        return SubAgent("test", TEST_SYSTEM,
                        ["run_tests", "read_file", "search_files"], llm=llm,
                        max_steps=max_steps, summary_limit=500)
    raise ValueError(f"未知子代理类型: {kind}")
