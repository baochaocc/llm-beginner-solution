import time

import requests

for name, url in {
    "ollama.com/download": "https://ollama.com/download/OllamaSetup.exe",
    "ollama.com/download windows": "https://ollama.com/download/windows",
}.items():
    t0 = time.time()
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        print(f"{name:28s} -> {r.status_code} final={r.url[:80]} len={r.headers.get('Content-Length')} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"{name:28s} -> EXC {type(e).__name__} {str(e)[:60]}")
