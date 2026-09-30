import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.indexer import build_index

build_index()
