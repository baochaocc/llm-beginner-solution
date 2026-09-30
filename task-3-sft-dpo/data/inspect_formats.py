import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root = Path(__file__).parent

print("=== MOSS 10w ===")
f = list((root / "moss-sft-10w").glob("*.jsonl"))[0]
line = f.read_text(encoding="utf-8").splitlines()[0]
obj = json.loads(line)
print("keys:", list(obj.keys())[:8])
print("sample:", json.dumps(obj, ensure_ascii=False)[:400])

print("\n=== Human-Like DPO ===")
for f in sorted((root / "dpo-humanlike").glob("*.json")):
    data = json.loads(f.read_text(encoding="utf-8"))
    n = len(data) if isinstance(data, list) else "?"
    first = data[0] if isinstance(data, list) else data
    keys = list(first.keys()) if isinstance(first, dict) else type(first)
    print(f"{f.name}: {n} 项 | keys={keys}")
    if isinstance(first, dict):
        print("   sample:", json.dumps(first, ensure_ascii=False)[:300])

print("\n=== ORCA DPO (parquet) ===")
p = list((root / "dpo-orca").glob("*.parquet"))[0]
df = pd.read_parquet(p)
print("rows:", len(df), "| cols:", list(df.columns)[:10])
print("sample:", df.iloc[0].to_dict())
