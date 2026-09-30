import requests
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def search(q, page=1):
    url = f"https://modelscope.cn/api/v1/dolphin/datasets?PageSize=15&PageNumber={page}&Query={requests.utils.quote(q)}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return []
    return r.json().get("Data") or []


for q in ["moss", "DPO", "preference", "alpaca", "chinese instruct"]:
    print(f"===== 搜索: {q} =====")
    for d in search(q):
        size = d.get("StorageSize", 0) / 1e6
        print(f"  {d.get('Namespace')}/{d.get('Name'):48s} {size:8.0f}MB  dl={d.get('Downloads')}")
