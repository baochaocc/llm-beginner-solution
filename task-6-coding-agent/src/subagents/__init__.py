"""任务六（M3 附加）：Subagent —— 独立 context / 独立步数上限 / 工具子集。

主 agent 只接收子代理的**摘要**（≤N 字），不吞全部 trace；子代理各自持独立的
message 列表，拥有自己的 mini-loop（模型 -> 工具 -> 观察 -> 结论）。
"""
from .worker import SubAgent, EXPLORE_SYSTEM, TEST_SYSTEM, make_subagent

__all__ = ["SubAgent", "EXPLORE_SYSTEM", "TEST_SYSTEM", "make_subagent"]
