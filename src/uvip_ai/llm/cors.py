"""CORS khusus path `/llm/*`.

Shim Bun lama (`server.mjs`) memasang `Access-Control-Allow-Origin: *` pada
semua responsnya, supaya prompt bisa di-`fetch()` dari halaman mana pun.
Router Python ini meniru perilaku itu, TAPI sengaja dibatasi hanya ke prefix
`/llm` — endpoint AI (upload/process video) tidak perlu dipanggil lintas-origin,
jadi jangan diperluas tanpa alasan.

Dipasang sebagai middleware app-level (Starlette hanya punya middleware di
level `app`, bukan `APIRouter`), lalu langsung `pass-through` untuk path lain.
"""

from __future__ import annotations

PREFIX = "/llm"

_CORS_HEADERS = (
    (b"access-control-allow-origin", b"*"),
    (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
    (b"access-control-allow-headers", b"*"),
    (b"access-control-max-age", b"86400"),
)


class LLMCorsMiddleware:
    """Tambah header CORS hanya untuk request ke `/llm*`."""

    def __init__(self, app, prefix: str = PREFIX):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith(self.prefix):
            return await self.app(scope, receive, send)

        # Preflight: jawab sendiri, jangan teruskan ke route (route GET-only
        # akan balas 405 untuk OPTIONS dan preflight dianggap gagal browser).
        if scope.get("method") == "OPTIONS":
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": list(_CORS_HEADERS),
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_with_cors(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.extend(_CORS_HEADERS)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_cors)


__all__ = ["LLMCorsMiddleware"]
