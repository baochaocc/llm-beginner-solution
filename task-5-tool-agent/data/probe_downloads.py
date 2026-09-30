import time

import requests

tests = {
    "registry.ollama.ai": "https://registry.ollama.ai/v2/library/qwen2.5:latest/manifests/7b-instruct",
    "ghfast.top installer": "https://ghfast.top/https://github.com/ollama/ollama/releases/download/v0.33.2/OllamaSetup.exe",
    "ghproxy.net installer": "https://ghproxy.net/https://github.com/ollama/ollama/releases/download/v0.33.2/OllamaSetup.exe",
    "mirror.ghproxy.com": "https://mirror.ghproxy.com/https://github.com/ollama/ollama/releases/download/v0.33.2/OllamaSetup.exe",
}
for name, url in tests.items():
    t0 = time.time()
    try:
        r = requests.head(url, timeout=8, allow_redirects=True)
        print(f"{name:28s} -> {r.status_code} len={r.headers.get('Content-Length')} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"{name:28s} -> EXC {type(e).__name__} {str(e)[:50]}")
