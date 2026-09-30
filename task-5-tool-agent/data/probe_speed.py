import time

import requests

URL = "https://release-assets.githubusercontent.com/github-production-release-asset/658b46d4dc4b8c903676fb12328f2cbd4d0fe1185e30fbb738a67b1dfdb9a0cf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH3Q%2F20260902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260902T000000Z&X-Amz-Expires=300&X-Amz-Signature=placeholder"
# 上面是占位；直接用它 HEAD 得到的最终 URL 会过期。改用 ollama.com 跳转流程并流式测速：
t0 = time.time()
r = requests.get("https://ollama.com/download/OllamaSetup.exe", stream=True, timeout=(30, 120))
print("status:", r.status_code, "| final host:", r.url.split("/")[2])
n = 0
for chunk in r.iter_content(1 << 20):
    n += len(chunk)
    if n >= 16 * (1 << 20):
        break
dt = time.time() - t0
print(f"16MB 用时 {dt:.1f}s -> {n / dt / 1e6:.2f} MB/s")
