"""下载脚本兼容替代：datasets>=4 已移除"脚本型数据集"支持，
seamew/ChnSentiCorp 的 ChnSentiCorp.py 无法再被 load_dataset 执行。

本脚本直接从该数据集仓库下载现成的 arrow 文件（与 loader 处理结果完全一致），
再用 pyarrow 读出来转成 train/validation/test.parquet。

用法：python data/download_arrow.py   （可选环境变量 HF_ENDPOINT=https://hf-mirror.com）
"""
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow.ipc as ipc

DATA_DIR = Path(__file__).parent
BASE = (os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        + "/datasets/seamew/ChnSentiCorp/resolve/main")
SPLITS = {
    "train": "chn_senti_corp-train.arrow",
    "validation": "chn_senti_corp-validation.arrow",
    "test": "chn_senti_corp-test.arrow",
}


def download(url: str, dest: Path):
    print(f"下载 {url} -> {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "dsh"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def read_arrow(path: Path):
    with open(path, "rb") as f:
        try:
            with ipc.open_file(f) as reader:      # IPC file 格式
                table = reader.read_all()
        except (ValueError, OSError):
            f.seek(0)
            with ipc.open_stream(f) as reader:    # IPC stream 格式
                table = reader.read_all()
    return table


def main():
    cache = DATA_DIR / "cache_dl"
    cache.mkdir(exist_ok=True)

    for split, fname in SPLITS.items():
        arrow_path = cache / fname
        if not arrow_path.exists():
            download(f"{BASE}/{fname}", arrow_path)
        table = read_arrow(arrow_path)
        out = DATA_DIR / f"{split}.parquet"
        table.to_pandas().to_parquet(str(out), index=False)
        print(f"  {split}: {table.num_rows} 行, 列 {table.column_names} -> {out.name}")

    print("\n完成。可直接运行 python train.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        print("提示：若直连 HF 失败，先运行: $env:HF_ENDPOINT='https://hf-mirror.com' 再重试", file=sys.stderr)
        sys.exit(1)
