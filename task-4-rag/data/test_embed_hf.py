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

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/bge-small-zh-v1.5")
model = AutoModel.from_pretrained("models/bge-small-zh-v1.5").cuda().eval()

t0 = time.time()
outs = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=512,
                  return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        out = model(**enc)[0]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        outs.append(F.normalize(pooled, p=2, dim=-1).float().cpu())
emb = torch.cat(outs).numpy()
print(f"HF 路径 encode OK: {emb.shape} ({time.time() - t0:.0f}s)")
