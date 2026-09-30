"""任务五 M1：wiki（维基百科查询，中文优先、英文兜底）。

⚠️ 本机网络环境下 zh/en.wikipedia.org 均被墙：查询会失败并抛出异常 →
自检把 wiki 归为「跳过」（网络不可用），其余工具不受影响；
Agent 端此时应依据自身知识作答（如图灵机发明者、Hinton 出生年份）。
"""
import json
import urllib.parse

import requests

TOOL_SCHEMA = {
    "name": "wiki",
    "description": "查询维基百科（先中文后英文），返回条目的摘要内容。"
                   "查询词用中文或英文均可。",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "要查询的词条"}},
        "required": ["query"],
    },
}

_APIS = [
    "https://zh.wikipedia.org/w/api.php",
    "https://en.wikipedia.org/w/api.php",
]


def _fetch(api, query, timeout=6):
    params = {
        "action": "query", "prop": "extracts", "exintro": 1,
        "explaintext": 1, "format": "json", "redirects": 1,
        "titles": query,
    }
    r = requests.get(api, params=params, timeout=timeout,
                     headers={"User-Agent": "nndl-llm-beginner/1.0"})
    r.raise_for_status()
    data = json.loads(r.text)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = (page.get("extract") or "").strip()
        if extract:
            return f"[{urllib.parse.urlparse(api).netloc}] {extract}"
    return ""


def run(args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("缺少 query 参数")
    errors = []
    for api in _APIS:
        try:
            out = _fetch(api, query)
            if out:
                return out
            errors.append(f"{api}: 无条目")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{api}: {type(e).__name__}")
    raise RuntimeError("无法访问维基百科（网络受限）: " + "; ".join(errors))
