import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.tools import calculator, python_sandbox, file_search, wiki

print("calc 2+3*4:", calculator.run({"expression": "2 + 3 * 4"}))
print("calc sqrt(2026):", calculator.run({"expression": "sqrt(2026)"}))
print("calc (123+456)*789:", calculator.run({"expression": "(123 + 456) * 789"}))
print("python sum(range(10)):", python_sandbox.run({"code": "print(sum(range(10)))"}))
print("python 质数和:",
      python_sandbox.run({"code": "print(sum(x for x in range(2,101) if all(x % i for i in range(2,x))))"}))
print("python 回文:",
      python_sandbox.run({"code": "s = 'level'; print(s, s == s[::-1]); t = 'world'; print(t, t == t[::-1])"}))
print("python 逃逸(应被拒):", python_sandbox.run({"code": "print(().__class__.__bases__)"})[:80])

root = os.getcwd()
out = file_search.run({"pattern": "README.md", "dir": root})
print("file_search contains README.md:", "README.md" in out)
print("  片段:", out[:150].replace("\n", " / "))
out2 = file_search.run({"pattern": "TODO", "dir": root})
print("file_search TODO -> todo_note.md:", "todo_note.md" in out2)
try:
    print("wiki:", wiki.run({"query": "Alan Turing"})[:100])
except Exception as e:
    print("wiki 失败(预期,被墙):", type(e).__name__, str(e)[:60])
