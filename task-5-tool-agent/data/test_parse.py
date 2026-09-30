"""解析器回归测试：复现 T6 中模型输出的原生 <tool_call> 形式。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import parse_reply

# 与 debug_t6_t9.py 截获的模型原始输出完全一致（\n 为 JSON 转义，非真实换行）
raw = (
    "<tool_call>\n"
    '{"name": "python_sandbox", "arguments": {"code": "def is_palindrome(s):\\n'
    '    s = str(s)\\n'
    '    return s == s[::-1]}\\n'
    "is_palindrome('level')\\n"
    "is_palindrome('world')\"}}\n"
    "</tool_call>"
)
print("RAW:", repr(raw))
print("PARSE:", parse_reply(raw))

# 变体 1：无换行的紧凑形式
v1 = '<tool_call>{"name":"python_sandbox","arguments":{"code":"print(1)"}}</tool_call>'
print("V1:", parse_reply(v1))

# 变体 2：带思考前缀
v2 = '让我用工具计算。<tool_call>{"name":"calculator","arguments":{"expression":"2+3*4"}}</tool_call>'
print("V2:", parse_reply(v2))

# 变体 3：裸 JSON
v3 = '{"name": "wiki", "arguments": {"query": "图灵机"}}'
print("V3:", parse_reply(v3))

# 变体 4：混入词法检查 <|tool_call|> 形式
v4 = '<|tool_call|>\n{"name": "calculator", "arguments": {"expression": "2+3*4"}}\n<|/tool_call|>'
print("V4:", parse_reply(v4))
