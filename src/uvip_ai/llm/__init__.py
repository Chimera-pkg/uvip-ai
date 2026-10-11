"""Gateway LLM (9router) untuk UVIP-AI.

Modul ini adalah port Python dari `server.mjs` (shim Bun) yang dulu berdiri
sendiri di port 8787. Sekarang menempel di API utama sebagai router `/llm/*`,
jadi ikut muncul di `/docs` dan tidak perlu proses terpisah.
"""

from uvip_ai.llm.router import router

__all__ = ["router"]
