import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.indexer import extract_pdf_pages
from src.chunker import chunk_documents
import time

pages = extract_pdf_pages("data/kb.pdf")
chunks = chunk_documents(pages, 512, 64)
texts = [c[0] for c in chunks]
print("chunks:", len(texts))

t0 = time.time()
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("models/bge-small-zh-v1.5", device="cuda")
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    print(f"ST(无进度条) encode OK: {emb.shape} ({time.time() - t0:.0f}s)")
except Exception as e:
    print(f"ST 失败: {type(e).__name__} {str(e)[:100]}")
