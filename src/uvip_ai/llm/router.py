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
    POST /llm/chat                 -> chat multi-turn (body JSON), balas JSON + context
    POST /llm/chat/completions     -> passthrough (drop-in base URL, stream jalan)
    POST /llm/vision               -> analisa gambar (multipart), balas JSON terstruktur

Catatan penting: upstream SELALU membalas `text/event-stream` walaupun isinya
satu objek JSON utuh (bukan SSE sungguhan) — lihat `_extract_json()`. Tanpa
penanganan itu, `JSON.parse()` gagal dan semua permintaan tampak error.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

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


# ─── Vision (POST /llm/vision) ───────────────────────────────────────────────

# Batas ukuran gambar. Base64 membengkakkan payload ~33%, dan proxy/Vast sering
# menolak body besar, jadi 8 MB mentah (~11 MB terkirim) adalah batas aman.
VISION_MAX_BYTES = 8 * 1024 * 1024
VISION_MAX_TOKENS_CAP = 4096

# Format yang diterima -> media type untuk data URL. Daftar putih (bukan sniff)
# supaya berkas "aneh" tidak diteruskan mentah ke upstream.
_VISION_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

def _vision_media_type(filename: Optional[str], content_type: Optional[str]) -> str:
    """Tentukan media type gambar dari nama file, fallback ke content-type."""
    ext = Path(filename or "").suffix.lower()
    if ext in _VISION_MEDIA_TYPES:
        return _VISION_MEDIA_TYPES[ext]
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return ct
    raise HTTPException(
        status_code=415,
        detail=("format gambar tidak didukung; pakai salah satu: "
                + ", ".join(sorted(_VISION_MEDIA_TYPES))),
    )

async def _read_vision_upload(file: UploadFile) -> bytes:
    """Baca upload dengan batas ukuran (tolak lebih awal, jangan buffer buta)."""
    data = await file.read(VISION_MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="file gambar kosong")
    if len(data) > VISION_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"gambar terlalu besar (maks {VISION_MAX_BYTES // (1024 * 1024)} MB)",
        )
    return data

def _vision_int(raw: Optional[str], *, name: str, default: int, lo: int, hi: int) -> int:
    if raw in (None, ""):
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"parameter '{name}' harus angka")
    return max(lo, min(hi, val))

def _vision_float(raw: Optional[str], *, name: str, default: Optional[float]) -> Optional[float]:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"parameter '{name}' harus angka")

async def _ask_vision(*, model: str, prompt: str, system: Optional[str],
                      image_bytes: bytes, media_type: str, max_tokens: int,
                      temperature: Optional[float], detail: str) -> dict:
    """Versi `_ask` untuk gambar: susun content multimodal, lalu fallback sama."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    messages: list = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": detail}},
        ],
    })
    # Model vision dipisah dari model teks: fallback teks belum tentu bisa
    # membaca gambar, jadi rantai di sini dibangun dari _fallbacks() yang sama
    # tapi model utamanya = NINE_ROUTER_VISION_MODEL.
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
            logger.warning("[llm] vision model=%s gagal: %s", m, exc.detail)
    assert last is not None
    raise last

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


# ─── Chat (POST /llm/chat) ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Satu giliran percakapan (bentuk OpenAI)."""
    role: str = Field(..., description="system | user | assistant")
    content: str = Field(..., description="Isi pesan")


class ChatRequest(BaseModel):
    """Body POST /llm/chat.

    `messages` = mode percakapan penuh (multi-turn). Kalau kosong, `q` dipakai
    sebagai satu pesan `user` (ditambah `system` bila diisi).
    """
    messages: Optional[list[ChatMessage]] = Field(
        None, description="Riwayat percakapan (multi-turn). Utamakan ini.")
    q: Optional[str] = Field(None, description="Pertanyaan singkat (single-turn).")
    system: Optional[str] = Field(None, description="System prompt.")
    model: Optional[str] = Field(None, description="Timpa model default.")
    max_tokens: Optional[int] = Field(1024, ge=1, le=32768)
    temperature: Optional[float] = Field(None, ge=0, le=2)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"q": "Jelaskan apa itu H.264 dengan singkat."},
                {
                    "system": "Jawab singkat dalam bahasa Indonesia.",
                    "messages": [
                        {"role": "user", "content": "Siapa presiden pertama Indonesia?"},
                        {"role": "assistant", "content": "Ir. Soekarno."},
                        {"role": "user", "content": "Berapa lama dia menjabat?"},
                    ],
                },
            ]
        }
    }


def _messages_from_body(req: ChatRequest) -> list:
    """Ubah body menjadi daftar `messages` untuk upstream (validasi + error jelas)."""
    if req.messages:
        out: list = []
        for m in req.messages:
            role = (m.role or "").strip().lower()
            if role not in ("system", "user", "assistant"):
                raise HTTPException(
                    status_code=400,
                    detail=f"role '{m.role}' tidak dikenal (pakai system|user|assistant)",
                )
            if not (m.content or "").strip():
                raise HTTPException(status_code=400, detail="'content' tidak boleh kosong")
            out.append({"role": role, "content": m.content})
        if req.system:
            out.insert(0, {"role": "system", "content": req.system})
        return out
    if (req.q or "").strip():
        return _build_messages(req.q, req.system, None)
    raise HTTPException(
        status_code=400, detail="isi 'q' atau 'messages' (minimal salah satu)")


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


@router.post("/chat", summary="Chat AI (multi-turn), balas JSON + context")
async def llm_chat(body: ChatRequest):
    """Endpoint chat utama: kirim riwayat percakapan, terima jawaban model.

    Bedanya dengan yang lain:
      - `POST` + body JSON (rapi & enak dites di /docs)
      - dukung **multi-turn** lewat `messages`
      - pakai rantai fallback model yang sama dengan `/llm/ask`

    Balasan = bentuk OpenAI + `answer` + `context` (model, waktu, pesan terkirim).
    """
    messages = _messages_from_body(body)
    model = (body.model or "").strip() or settings.nine_router_model
    out = await _ask(model=model, messages=messages,
                     max_tokens=body.max_tokens or 1024,
                     temperature=body.temperature)
    logger.info("POST /llm/chat model=%s %dms turns=%d len=%d",
                out["model_used"], out["elapsed_ms"], len(messages), len(out["content"]))
    raw = out["raw"]
    return {
        "id": raw.get("id") or f"chatcmpl-chat-{int(time.time() * 1000)}",
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
            "requested_model": out["requested_model"],
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


@router.post("/vision", summary="Analisa gambar pakai LLM vision, balas JSON terstruktur")
async def llm_vision(
    file: UploadFile = File(..., description="Berkas gambar: jpg/jpeg/png/webp/gif/bmp"),
    prompt: str = Form("Jelaskan isi gambar ini secara singkat."),
    system: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    detail: str = Form("auto"),
):
    """Kirim gambar + prompt ke LLM vision, balas JSON terstruktur.

    `detail` = auto|low|high (petunjuk ke upstream seberapa detail gambar dibaca).
    Model default dari `NINE_ROUTER_VISION_MODEL`; `model` di sini menimpanya.
    """
    media_type = _vision_media_type(file.filename, file.content_type)
    image_bytes = await _read_vision_upload(file)
    if not (prompt or "").strip():
        raise HTTPException(status_code=400, detail="parameter 'prompt' wajib diisi")
    if detail not in ("auto", "low", "high"):
        raise HTTPException(status_code=400, detail="parameter 'detail' harus auto|low|high")

    chosen = (model or "").strip() or settings.nine_router_vision_model
    out = await _ask_vision(
        model=chosen,
        prompt=prompt,
        system=system,
        image_bytes=image_bytes,
        media_type=media_type,
        max_tokens=_vision_int(max_tokens, name="max_tokens",
                               default=1024, lo=1, hi=VISION_MAX_TOKENS_CAP),
        temperature=_vision_float(temperature, name="temperature", default=None),
        detail=detail,
    )
    logger.info("POST /llm/vision model=%s %dms bytes=%d len=%d",
                out["model_used"], out["elapsed_ms"], len(image_bytes), len(out["content"]))
    raw = out["raw"]
    return {
        "id": raw.get("id") or f"chatcmpl-vision-{int(time.time() * 1000)}",
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
            "requested_model": out["requested_model"],
            "elapsed_ms": out["elapsed_ms"],
            "image": {
                "filename": file.filename,
                "media_type": media_type,
                "bytes": len(image_bytes),
                "detail": detail,
            },
            "prompt": prompt,
        },
        "usage": raw.get("usage"),
    }

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
