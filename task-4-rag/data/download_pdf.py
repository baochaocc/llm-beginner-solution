"""下载 NNDL v2 PDF（多候选源 + 重试）。"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests

DEST = Path(__file__).resolve().parents[1] / "data" / "kb.pdf"
URLS = [
    "https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf",
    "https://ghproxy.net/https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf",
    "https://ghfast.top/https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf",
    "https://gh-proxy.com/https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf",
]


def main():
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"已存在：{DEST} ({DEST.stat().st_size / 1e6:.0f}MB)")
        return
    for url in URLS:
        for attempt in range(3):
            try:
                print(f"尝试 {url}（第 {attempt + 1} 次）...")
                with requests.get(url, stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0) or 0)
                    done = 0
                    with DEST.open("wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                    if total and done != total:
                        raise IOError(f"长度不符 {done}/{total}")
                    print(f"完成：{DEST} ({done / 1e6:.0f}MB)")
                    return
            except Exception as e:
                print(f"  失败：{type(e).__name__} {str(e)[:80]}")
                time.sleep(5)
    sys.exit("所有源均失败")


if __name__ == "__main__":
    main()
