"""M1 附加验证：MCP stdio 协议握手 + list_tools（README 要求可独立启动）。"""
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "src" / "mcp_server.py"), "--repo", str(ROOT / "data" / "toy-repo")],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("MCP stdio handshake OK, tools:", [t.name for t in tools.tools])
            r = await session.call_tool("run_tests", {})
            print("call run_tests ->", str(r.content[0].text)[:120].replace("\n", " | "))


asyncio.run(main())
