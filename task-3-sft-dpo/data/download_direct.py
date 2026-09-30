"""直接下载 MOSS SFT / DPO 数据（绕开 huggingface_hub 的元数据接口）。

原因：本环境下 huggingface.co 直连被重置、hf-mirror 的 hub 元数据接口
偶发 RepositoryNotFoundError，但 resolve 文件地址是通的。
用法：python data/download_direct.py
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests

MIRROR = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ("datasets/OpenMOSS-Team/moss-003-sft-data/resolve/main/moss-003-sft-no-tools.jsonl.zip",
     ROOT / "data" / "moss-sft" / "moss-003-sft-no-tools.jsonl.zip"),
    ("datasets/hiyouga/DPO-En-Zh-20k/resolve/main/dpo_zh.json",
     ROOT / "data" / "dpo" / "dpo_zh.json"),
    ("datasets/hiyouga/DPO-En-Zh-20k/resolve/main/dpo_en.json",
     ROOT / "data" / "dpo" / "dpo_en.json"),
]


def download(url, dst, max_retry=6):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(max_retry):
        try:
            print(f"下载 {url.split('/datasets/')[1]} ...")
            with requests.get(url, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0) or 0)
                done = 0
                with dst.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                if total and done != total:
                    raise IOError(f"长度不符 {done}/{total}")
                print(f"  完成：{dst.name} ({done / 1e6:.0f}MB)")
                return True
        except Exception as e:
            print(f"  重试 {i + 1}/{max_retry}：{type(e).__name__} {str(e)[:80]}")
            time.sleep(10 * (i + 1))
    return False


def main():
    ok = True
    for path, dst in FILES:
        if dst.exists() and dst.stat().st_size > 0:
            print(f"已存在，跳过：{dst.name}")
            continue
        ok &= download(f"{MIRROR}/{path}", dst)
    print("全部完成" if ok else "存在失败文件，请重跑本脚本")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
