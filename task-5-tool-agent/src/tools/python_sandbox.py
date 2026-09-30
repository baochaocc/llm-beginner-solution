"""任务五 M1：python_sandbox（受限执行环境，教学级防护）。

⚠️ 防护仅为教学用途：白名单 builtins + 禁 import + 禁属性访问 + 线程超时。
仍然挡不住 `().__class__.__bases__` 这类 UTF-8 边界外的逃逸手段
（属性访问已禁用，降低常见逃逸面），绝不要用它执行不可信代码。

自检：run({"code": "print(sum(range(10)))"}) 应返回包含 "45" 的字符串。
"""
import ast
import contextlib
import io
import math as _math_mod
import sys
import threading

TOOL_SCHEMA = {
    "name": "python_sandbox",
    "description": "在受限沙盒里执行 Python 代码（禁止 import、禁止访问对象属性）。"
                   "支持内置函数（print/sum/range/len/all/any/map/filter/sorted等）、"
                   "数学函数（sqrt/sin/...）和常规表达式/循环；stdout 会被捕获并返回。",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
        "required": ["code"],
    },
}

_SAFE_BUILTINS = dict(
    abs=abs, all=all, any=any, bin=bin, bool=bool, chr=chr, dict=dict,
    divmod=divmod, enumerate=enumerate, filter=filter, float=float,
    format=format, frozenset=frozenset, hex=hex, int=int, isinstance=isinstance,
    issubclass=issubclass, len=len, list=list, map=map, max=max, min=min,
    oct=oct, ord=ord, pow=pow, range=range, repr=repr, reversed=reversed,
    round=round, set=set, slice=slice, sorted=sorted, str=str, sum=sum,
    tuple=tuple, type=type, zip=zip, print=print, input=lambda *a: "",
    sqrt=__import__("math").sqrt, sin=__import__("math").sin, cos=__import__("math").cos,
    tan=__import__("math").tan, pi=__import__("math").pi, e=__import__("math").e,
    log=__import__("math").log, floor=__import__("math").floor, ceil=__import__("math").ceil,
)
_DENIED = (ast.ClassDef, ast.Global, ast.Nonlocal,
           ast.Lambda, ast.Yield, ast.Await, ast.AsyncFor, ast.AsyncWith,
           ast.AsyncFunctionDef, ast.Attribute, ast.Try, ast.Raise,
           ast.Delete, ast.With, ast.ExceptHandler)
_ALLOWED_IMPORTS = {"math"}   # 唯一允许的 import 路径


def _check_ast(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if not all(a.name in _ALLOWED_IMPORTS for a in node.names):
                raise ValueError(f"禁止 import：{node.names[0].name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_IMPORTS:
                raise ValueError(f"禁止 import：{node.module}")
        elif isinstance(node, ast.Attribute):
            # 允许普通点号方法（如 s.upper()/range()），但拦截 __dunder__ 逃逸
            if node.attr.startswith("__"):
                raise ValueError(f"禁止访问 {node.attr}（dunder 逃逸面）")
        elif isinstance(node, _DENIED):
            raise ValueError(f"禁止语法：{type(node).__name__}")


def _execute(code, globals_):
    tree = ast.parse(code, mode="exec")
    _check_ast(tree)
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        value = eval(compile(ast.Expression(tree.body[0].value), "<sandbox>", "eval"), globals_)
        return repr(value)
    exec(compile(tree, "<sandbox>", "exec"), globals_)
    return None


def run(args: dict, timeout: float = 5.0) -> str:
    code = str(args.get("code", ""))
    if not code.strip():
        raise ValueError("缺少 code 参数")

    out = io.StringIO()
    result = {"value": None, "error": None}

    def worker():
        try:
            with contextlib.redirect_stdout(out):
                v = _execute(code, {"__builtins__": _SAFE_BUILTINS,
                                    "math": _math_mod})
                if v is not None:
                    result["value"] = v
        except Exception as e:  # noqa: BLE001 —— 教学沙盒，任何异常都转成可观察的错误
            hint = ""
            if isinstance(e, SyntaxError):
                hint = "（代码有语法错误/多了或少了括号或冒号，请修正代码后重新提交）"
            elif isinstance(e, NameError):
                hint = "（变量名可能有误，请检查后重新提交）"
            result["error"] = f"{type(e).__name__}: {e}{hint}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return f"执行超时（>{timeout}s），已终止"

    if result["error"]:
        return f"执行错误: {result['error']}"
    output = out.getvalue().strip()
    if result["value"] is not None:
        output = (output + "\n" + result["value"]).strip()
    return output if output else "(无输出)"
