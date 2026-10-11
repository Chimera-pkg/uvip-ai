"""Router LLM (9router) — port Python dari `llm/server.mjs`.

Kenapa ada varian GET?
    Upstream 9router itu OpenAI-compatible tapi HANYA menerima POST
    (`GET /v1/chat/completions` -> 405). Router ini menambah endpoint GET
    supaya prompt bisa dipanggil langsung dari browser, `<iframe>`, `<img>`,
    `curl`, atau `fetch` tanpa perlu JS tambahan.

Endpoint (semua ikut muncul di /docs):
    GET  /llm/ask                  -> text/plain, jawaban mentah
    GET  /llm/chat/completions     -> JSON OpenAI-shape + context
    GET  /llm/models               -> daftar model upstream
    GET  /llm/health               -> status upstream
    POST /llm/chat/completions     -> passthrough (drop-in base URL, stream jalan)

Catatan penting: upstream SELALU membalas `text/event-stream` walaupun isinya
satu objek JSON utuh (bukan SSE sungguhan) — lihat `_extract_json()`. Tanpa
penanganan itu, `JSON.parse()` gagal dan semua permintaan tampak error.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from uvip_ai.config import settings

logger = logging.getLogger("uvip_ai.llm")

router = APIRouter(prefix="/llm", tags=["LLM (9router)"])

# Halaman uji embed — file terpisah supaya bisa diedit tanpa menyentuh Python.
DEMO_HTML_PATH = Path(__file__).with_name("demo.html")


# ─── Helper ─────────────────────────────────────────────────────────────────

def _base_url() -> str:
    """URL upstream tanpa slash di ujung."""
    return (settings.nine_router_url or "").rstrip("/")


def _fallbacks() -> list:
    """Daftar model fallback (dipisah koma), urut sesuai konfigurasi."""
    raw = settings.nine_router_fallbacks or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if settings.nine_router_key:
        h["Authorization"] = f"Bearer {settings.nine_router_key}"
    return h


def _extract_json(text: str) -> dict:
    """Ambil objek JSON dari balasan upstream.

    Upstream mengirim `Content-Type: text/event-stream` tapi isinya cuma satu
    objek JSON utuh diikuti `data: [DONE]` (bukan SSE berlapis). Karena itu
    `json.loads(text)` mentah selalu gagal ("Extra data"). Di sini kita:
      1. coba parse utuh;
      2. kalau gagal, ambil baris `data: {...}` yang valid;
      3. kalau tetap gagal, coba objek JSON pertama lewat raw_decode.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Bentuk SSE: baris-baris "data: {...}"
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    # JSON pertama yang bisa di-decode (abaikan ekor "data: [DONE]").
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {"error": {"message": text[:500]}}


def _is_retryable(msg: str) -> bool:
    """Error yang pantas dicoba ke model fallback.

    Disamakan dengan regex di server.mjs, tapi TANPA memakai `re` karena
    polanya sederhana — cukup cek substring huruf kecil.
    """
    m = (msg or "").lower()
    keys = ("blocked", "not found", "unavailable", "reset after",
            "rate limit", "ratelimit", "429", "500", "502", "503", "504")
    return any(k in m for k in keys)


async def _call_upstream(*, model: str, messages: list, max_tokens: int,
                         temperature: Optional[float], extra: Optional[dict] = None) -> dict:
    """Panggil upstream /chat/completions (non-stream). Raise HTTPException bila gagal."""
    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                  # WAJIB false: tanpa ini upstream membalas event-stream berisi
                  # JSON + "data: [DONE]" yang bikin parsing gagal.
                  "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if extra:
        body.update(extra)

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=settings.nine_router_timeout_s) as client:
            res = await client.post(f"{_base_url()}/chat/completions",
                                    headers=_headers(), json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502,
                            detail=f"upstream tidak terjangkau: {type(exc).__name__}: {exc}")

    data = _extract_json(res.text)
    elapsed_ms = int((time.time() - t0) * 1000)

    if res.status_code >= 400 or data.get("error"):
        msg = (data.get("error") or {}).get("message") or f"HTTP {res.status_code}"
        raise HTTPException(status_code=res.status_code if res.status_code >= 400 else 502,
                            detail=msg)

    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    return {
        "raw": data,
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "elapsed_ms": elapsed_ms,
        "model_used": data.get("model") or model,
    }


async def _ask(*, model: str, messages: list, max_tokens: int,
               temperature: Optional[float]) -> dict:
    """Coba model utama, lalu fallback berurutan bila errornya retryable."""
    # Dedupe tapi pertahankan urutan (dict sebagai ordered set).
    chain = list(dict.fromkeys([model, *_fallbacks()]))
    last: Optional[HTTPException] = None
    for m in chain:
        try:
            out = await _call_upstream(model=m, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature)
            out["requested_model"] = model
            return out
        except HTTPException as exc:
            last = exc
            logger.warning("[llm] model=%s gagal: %s", m, exc.detail)
    assert last is not None
    raise last


def _build_messages(q: str, system: Optional[str], history: Optional[list]) -> list:
    msgs: list = []
    if system:
        msgs.append({"role": "system", "content": system})
    for h in history or []:
        if isinstance(h, dict) and h.get("role") and isinstance(h.get("content"), str):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": q})
    return msgs


def _parse_history(raw: Optional[str]) -> Optional[list]:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="parameter 'history' harus JSON array")
    if not isinstance(val, list):
        raise HTTPException(status_code=400, detail="parameter 'history' harus JSON array")
    return val


def _common_params(request: Request) -> dict:
    """Ambil parameter GET yang dipakai /llm/ask dan /llm/chat/completions."""
    p = request.query_params
    q = p.get("q") or p.get("prompt") or p.get("query") or p.get("message")
    if not q:
        raise HTTPException(status_code=400, detail="parameter 'q' wajib diisi")
    temp_raw = p.get("temperature")
    try:
        max_tokens = int(p.get("max_tokens") or 1024)
    except ValueError:
        raise HTTPException(status_code=400, detail="parameter 'max_tokens' harus angka")
    try:
        temperature = float(temp_raw) if temp_raw not in (None, "") else None
    except ValueError:
        raise HTTPException(status_code=400, detail="parameter 'temperature' harus angka")
    return {
        "q": q,
        "model": p.get("model") or settings.nine_router_model,
        "system": p.get("system"),
        "history": _parse_history(p.get("history")),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


# ─── Endpoint ───────────────────────────────────────────────────────────────

@router.get("/health", summary="Status upstream LLM")
async def llm_health():
    """Cek apakah upstream 9router hidup (GET /models)."""
    ok = False
    detail = ""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{_base_url()}/models", headers=_headers())
        ok = r.is_success
        detail = f"HTTP {r.status_code}"
    except httpx.HTTPError as exc:
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "ok": ok,
        "upstream": _base_url(),
        "detail": detail,
        "default_model": settings.nine_router_model,
        "fallbacks": _fallbacks(),
        "key_configured": bool(settings.nine_router_key),
    }


@router.get("/models", summary="Daftar model upstream")
async def llm_models():
    """Teruskan GET /models upstream apa adanya."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{_base_url()}/models", headers=_headers())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502,
                            detail=f"upstream tidak terjangkau: {type(exc).__name__}: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    return Response(content=r.content, media_type="application/json")


@router.get("/ask", summary="Tanya LLM, balas teks mentah",
            response_class=PlainTextResponse)
async def llm_ask(request: Request):
    """Jawaban model sebagai `text/plain` — paling enak untuk `<iframe>`/`curl`."""
    prm = _common_params(request)
    out = await _ask(model=prm["model"],
                     messages=_build_messages(prm["q"], prm["system"], prm["history"]),
                     max_tokens=prm["max_tokens"], temperature=prm["temperature"])
    logger.info("GET /llm/ask model=%s %dms len=%d",
                out["model_used"], out["elapsed_ms"], len(out["content"]))
    return PlainTextResponse(out["content"])


@router.get("/chat/completions", summary="Tanya LLM via GET, balas JSON OpenAI-shape")
async def llm_chat_get(request: Request):
    """Sama seperti `/llm/ask`, tapi balasannya JSON bentuk OpenAI + `context`."""
    prm = _common_params(request)
    messages = _build_messages(prm["q"], prm["system"], prm["history"])
    out = await _ask(model=prm["model"], messages=messages,
                     max_tokens=prm["max_tokens"], temperature=prm["temperature"])
    logger.info("GET /llm/chat/completions model=%s %dms len=%d",
                out["model_used"], out["elapsed_ms"], len(out["content"]))
    raw = out["raw"]
    return {
        "id": raw.get("id") or f"chatcmpl-get-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": out["model_used"],
        "requested_model": out["requested_model"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": out["content"],
                **({"reasoning_content": out["reasoning"]} if out["reasoning"] else {}),
            },
            "finish_reason": "stop",
        }],
        "answer": out["content"],
        "context": {
            "upstream": _base_url(),
            "model": out["model_used"],
            "elapsed_ms": out["elapsed_ms"],
            "messages": messages,
        },
        "usage": raw.get("usage"),
    }


@router.post("/chat/completions", summary="Passthrough ke upstream (drop-in base URL)")
async def llm_chat_post(request: Request):
    """Teruskan POST apa adanya (termasuk `stream: true`).

    Dipakai kalau ada klien yang cuma perlu base URL `/llm` sebagai pengganti
    `https://9router.../v1`.
    """
    body = await request.body()
    try:
        client = httpx.AsyncClient(timeout=settings.nine_router_timeout_s)
        upstream = await client.send(
            client.build_request("POST", f"{_base_url()}/chat/completions",
                                 headers=_headers(), content=body),
            stream=True,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502,
                            detail=f"upstream tidak terjangkau: {type(exc).__name__}: {exc}")

    async def _iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        _iter(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def llm_index():
    """Halaman uji embed (same-origin). Isinya `demo.html` di folder ini."""
    try:
        html = DEMO_HTML_PATH.read_text(encoding="utf-8")
    except OSError as exc:  # file hilang saat deploy -> jangan 500 tanpa jejak
        logger.error("demo.html tidak terbaca: %s", exc)
        raise HTTPException(500, f"demo.html tidak terbaca: {exc}") from exc
    return HTMLResponse(html)
