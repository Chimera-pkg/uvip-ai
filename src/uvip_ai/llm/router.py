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
    POST /llm/advise               -> foto + metrik UVI -> rekomendasi perbaikan
                                      (3 peran) + prompt gambar. Balas JSON terstruktur.

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


def _extract_json_object(text: str) -> Optional[dict]:
    """Ambil objek JSON pertama dari teks balasan LLM.

    Dipakai endpoint yang MEMINTA JSON ke model (mis. /llm/advise). Model sering
    membungkus JSON dengan pagar markdown (```json ... ```) atau menambah kalimat
    pembuka, sehingga ``json.loads`` mentah gagal. Fungsi ini tahan terhadap itu:
      1. buang pagar markdown;
      2. coba parse utuh;
      3. kalau gagal, cari '{' pertama dan ``raw_decode`` dari situ.
    Kembalikan ``None`` bila memang tidak ada objek JSON.
    """
    if not text:
        return None
    t = text.strip()
    # Buang pagar markdown ```json ... ``` (kalau ada).
    if t.startswith("```"):
        body = t[3:]
        end = body.find("```")
        if end != -1:
            body = body[:end]
        if body[:4].lower() == "json":
            body = body[4:]
        t = body.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Cari '{' pertama yang bisa di-decode (lewati kurung yang bukan JSON).
    idx = t.find("{")
    while idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(t[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = t.find("{", idx + 1)
    return None

def _is_retryable(msg: str) -> bool:
    """Error yang pantas dicoba ke model fallback.

    Disamakan dengan regex di server.mjs, tapi TANPA memakai `re` karena
    polanya sederhana — cukup cek substring huruf kecil.
    """
    m = (msg or "").lower()
    keys = ("blocked", "not found", "unavailable", "reset after",
            "rate limit", "ratelimit", "429", "500", "502", "503", "504",
            # ReadTimeout/ConnectTimeout -> model menggantung, pantas dicoba fallback.
            "timeout")
    return any(k in m for k in keys)


async def _call_upstream(*, model: str, messages: list, max_tokens: int,
                         temperature: Optional[float], extra: Optional[dict] = None,
                         timeout_s: Optional[float] = None) -> dict:
    """Panggil upstream /chat/completions (non-stream). Raise HTTPException bila gagal.

    ``timeout_s`` menimpa timeout global - dipakai /llm/advise supaya model yang
    menggantung (mis. glm-5.3-flash >120s pada prompt tanpa metrik) cepat
    menyerah dan rantai berlanjut ke fallback.
    """
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
        async with httpx.AsyncClient(timeout=timeout_s or settings.nine_router_timeout_s) as client:
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
    # Upstream kadang membalas model="unknown" (nama internal tidak diteruskan).
    # Simpan nama yang kita MINTA supaya log/context tetap informatif.
    used = data.get("model") or ""
    if not used or used.lower() == "unknown":
        used = model
    return {
        "raw": data,
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "elapsed_ms": elapsed_ms,
        "model_used": used,
        # finish_reason asli upstream: "length" menandakan balasan TERPOTONG
        # (max_tokens kurang). Endpoint yang meminta JSON wajib memeriksa ini.
        "finish_reason": choices[0].get("finish_reason") or "",
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

# ─── Advise (POST /llm/advise) ──────────────────────────────────────────────

ADVISE_MAX_TOKENS_CAP = 8192

# Kunci yang WAJIB ada di objek advice. Model fallback yang lebih lemah kadang
# mengarang skema sendiri (mis. "nilai_kondisi"/"temuan"), jadi kita validasi -
# JSON valid tapi skema salah tetap dianggap gagal dan dicoba ulang.
ADVISE_REQUIRED_KEYS = (
    "diagnosis", "masalah", "rekomendasi", "prompt_gambar", "before_after", "caveat",
)

# Metrik yang diharapkan datang dari segmentasi (jalur FOTO: seg.infer(excel_mode=True)).
# Semua opsional - endpoint tetap jalan tanpa metrik (analisa murni dari foto).
ADVISE_METRIC_KEYS = (
    "building_visibility_index",
    "vegetation_coverage_index",
    "sky_openness_index",
    "ground_accessibility_index",
    "human_activity_index",
    "vehicle_intensity_index",
    "traffic_infrastructure_index",
    "heritage_dominance_index",
)

# System prompt domain. Ditulis eksplisit supaya output konsisten JSON dan
# rekomendasinya spesifik (mengutip angka), bukan saran generik "perbanyak pohon".
_ADVISE_SYSTEM = """Anda analis ruang publik perkotaan Indonesia (jalan, trotoar, alun-alun).
Anda menerima FOTO + metrik kuantitatif hasil segmentasi semantik (persentase kelas
dan indeks 0..1) + 4 skor persepsi (beauty/safety/comfort/uvi, skala 1..10 dan 0..1).

TUGAS: baca foto, kaitkan dengan angkanya, lalu beri rekomendasi perbaikan yang
SPESIFIK dan bisa dieksekusi oleh 3 peran: perencana kota, insinyur sipil, arsitek.

ATURAN KETAT
1. Balas HANYA satu objek JSON valid. Tanpa kalimat pembuka, tanpa pagar markdown.
2. Setiap temuan WAJIB mengutip angka dari metrik yang diberikan. Jangan mengarang angka.
3. Skor tinggi (mis. 10/10) BUKAN berarti tidak ada masalah. Kalau foto menunjukkan
   masalah (parkir liar, trotoar terputus, kabel semrawut) sementara skor tinggi,
   tulis ketidaksesuaian itu di "diagnosis".
4. Rekomendasi harus konkret: sebut elemen apa, di sisi mana, ukuran/standar bila ada.
5. "standar" diisi acuan teknis Indonesia bila yakin (mis. Permen PUPR 03/2014 tentang
   Pedoman Perencanaan Penyediaan dan Pemanfaatan Prasarana dan Sarana Jaringan Pejalan
   Kaki; SNI 03-6967-2003 tentang Trotoar). Kalau tidak yakin, tulis "" (string kosong).
6. "prompt_gambar" = prompt BAHASA INGGRIS untuk text-to-image (Midjourney/SDXL),
   memuat: jenis ruang, material, vegetasi, pencahayaan, sudut kamera, dan gaya.
   Tuliskan kondisi IDEAL (perbaikan sudah dilakukan), bukan kondisi saat ini.
7. "caveat" wajib diisi: sebutkan bahwa analisa hanya dari 1 foto (tanpa data 3D,
   topografi, volume lalu lintas, atau survei lapangan) sehingga rekomendasi masih
   hipotesis yang perlu verifikasi lapangan.
8. Kalau metrik tidak diberikan, nilai murni dari foto dan tulis itu di "caveat".
9. RINGKAS - ini penting supaya balasan tidak terpotong:
   - "diagnosis": maksimal 3 kalimat.
   - "masalah": maksimal 4 item, tiap "temuan" maksimal 25 kata.
   - "rekomendasi": tepat 3 item (satu per peran: perencana_kota, insinyur_sipil, arsitek),
     tiap "tindakan" maksimal 40 kata.
   - "prompt_gambar": satu paragraf, maksimal 70 kata.
   - Jangan mengulang isi field di field lain.

SKEMA (semua field wajib ada; isi [] bila memang tidak ada):
{
  "diagnosis": "2-3 kalimat: kenapa kondisinya begini, kaitkan dengan angka",
  "masalah": [
    {"area": "trotoar", "temuan": "...", "bukti_angka": "walkability_ratio 0.024", "tingkat": "tinggi|sedang|rendah"}
  ],
  "rekomendasi": [
    {"peran": "perencana_kota", "tindakan": "...", "standar": "...", "dampak": "menaikkan safety_score", "prioritas": 1}
  ],
  "prompt_gambar": "English text-to-image prompt describing the improved street",
  "before_after": "narasi singkat perubahan yang diharapkan",
  "caveat": "..."
}
"""

def _advice_schema_ok(obj: dict) -> tuple[bool, str]:
    """Cek apakah objek advice memenuhi skema minimum.

    Hanya JSON valid TIDAK cukup: model fallback yang lebih lemah kadang
    mengarang skema sendiri (mis. daftar string, bukan objek ber-`peran`).
    Kembalikan (ok, alasan_gagal).
    """
    missing = [k for k in ADVISE_REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"kunci hilang: {', '.join(missing)}"
    if not isinstance(obj.get("diagnosis"), str) or not obj["diagnosis"].strip():
        return False, "'diagnosis' harus string tidak kosong"
    for key in ("prompt_gambar", "caveat", "before_after"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            return False, f"'{key}' harus string tidak kosong"
    if not isinstance(obj.get("masalah"), list):
        return False, "'masalah' harus array"
    recs = obj.get("rekomendasi")
    if not isinstance(recs, list) or not recs:
        return False, "'rekomendasi' harus array tidak kosong"
    # Tiap rekomendasi WAJIB objek ber-`peran` - inilah yang membuat sarannya
    # bisa dipisah per disiplin (perencana kota / insinyur sipil / arsitek).
    for i, item in enumerate(recs):
        if not isinstance(item, dict):
            return False, f"rekomendasi[{i}] harus objek, bukan {type(item).__name__}"
        for k in ("peran", "tindakan"):
            if not str(item.get(k) or "").strip():
                return False, f"rekomendasi[{i}].{k} wajib diisi"
    return True, ""

def _advise_chain(model: str) -> list:
    """Rantai model untuk /llm/advise: model utama lalu fallback vision-capable.

    Fallback teks biasa belum tentu bisa membaca gambar, jadi rantai di sini
    memakai `NINE_ROUTER_VISION_MODEL` sebagai cadangan pertama (bukan
    `NINE_ROUTER_MODEL`), baru `_fallbacks()`.
    """
    return list(dict.fromkeys([model, settings.nine_router_vision_model, *_fallbacks()]))

def _advise_prompt(*, metrics: Optional[dict], scores: Optional[dict],
                   location: Optional[str], extra: Optional[str]) -> str:
    """Susun pesan user: metrik + skor + lokasi sebagai konteks teks untuk LLM.

    Angka inilah yang membuat rekomendasi spesifik ("trotoar hanya 2,4% dari
    lebar jalan") alih-alih generik ("perbanyak trotoar").
    """
    lines = ["Analisa foto ruang publik berikut dan berikan rekomendasi perbaikan."]
    if location:
        lines.append(f"Lokasi: {location}")
    if metrics:
        lines.append("\nMetrik segmentasi (persentase kelas 0..100 & indeks 0..1):")
        for k in ADVISE_METRIC_KEYS:
            if k in metrics:
                lines.append(f"  - {k}: {metrics[k]}")
        # Sisanya (kalau ada) tetap ditampilkan supaya tidak ada info terbuang.
        for k, v in metrics.items():
            if k not in ADVISE_METRIC_KEYS and not str(k).startswith("_"):
                lines.append(f"  - {k}: {v}")
    if scores:
        lines.append("\nSkor persepsi (1..10 kecuali uvi 0..1):")
        for k, v in scores.items():
            if not str(k).startswith("_"):
                lines.append(f"  - {k}: {v}")
    if extra:
        lines.append(f"\nKonteks tambahan: {extra}")
    if not metrics and not scores:
        lines.append("\n(Metrik tidak tersedia - nilai kondisi murni dari foto.)")
    lines.append("\nBalas HANYA objek JSON sesuai skema di system prompt.")
    return "\n".join(lines)

async def _ask_advise(*, model: str, prompt: str, image_bytes: bytes,
                      media_type: str, max_tokens: int,
                      temperature: Optional[float], detail: str) -> dict:
    """Panggil LLM vision untuk analisa + rekomendasi, lalu parse JSON-nya.

    Beda dengan `_ask_vision`: di sini balasan WAJIB berupa JSON. Kalau model
    pertama membalas teks bebas, kita minta sekali lagi (repair) ke model yang
    sama sebelum pindah ke model fallback berikutnya.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    messages: list = [
        {"role": "system", "content": _ADVISE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": detail}},
            ],
        },
    ]
    last: Optional[HTTPException] = None
    last_parse_error = ""
    truncated = False
    for idx, m in enumerate(_advise_chain(model)):
        # Model pertama diberi waktu penuh; fallback diberi jatah lebih pendek
        # supaya total respons tidak menumpuk jadi berjam-jam saat semua menggantung.
        per_model_timeout = settings.nine_router_timeout_s if idx == 0 else min(
            90, settings.nine_router_timeout_s)
        # Percobaan repair menambah pesan ke `messages`; kembalikan ke kondisi
        # awal sebelum mencoba model berikutnya agar tidak menumpuk.
        base_len = len(messages)
        for attempt in (1, 2):
            try:
                out = await _call_upstream(model=m, messages=messages,
                                           max_tokens=max_tokens, temperature=temperature,
                                           timeout_s=per_model_timeout)
            except HTTPException as exc:
                last = exc
                logger.warning("[llm] advise model=%s gagal: %s", m, exc.detail)
                break  # model ini tidak bisa dipakai -> lanjut fallback
            finish = out.get("finish_reason") or ""
            parsed = _extract_json_object(out["content"])
            ok, why = (False, "tidak ada objek JSON")
            if parsed is not None:
                ok, why = _advice_schema_ok(parsed)
            # Balasan "length" = terpotong -> JSON tidak mungkin lengkap. Perlakukan
            # sebagai gagal-parse supaya dicoba ulang, tapi catat penyebabnya.
            if ok and finish != "length":
                out["advice"] = parsed
                out["requested_model"] = model
                out["parse_ok"] = True
                out["repair_used"] = attempt == 2
                out["truncated"] = False
                return out
            truncated = truncated or finish == "length"
            last_parse_error = out["content"][:300]
            logger.warning("[llm] advise model=%s gagal validasi (attempt %d, finish=%s): %s",
                           m, attempt, finish or "?", why)
            if attempt == 1:
                # Repair: minta ulang, tegas, tanpa gambar lagi (hemat token).
                messages.append({"role": "assistant", "content": out["content"][:2000]})
                messages.append({
                    "role": "user",
                    "content": (f"Balasan tadi tidak sesuai skema ({why}). Kirim ULANG HANYA "
                                "satu objek JSON dengan bentuk PERSIS begini (tipe harus sama):\n"
                                '{"diagnosis":"teks","masalah":[{"area":"teks","temuan":"teks",'
                                '"bukti_angka":"teks","tingkat":"tinggi|sedang|rendah"}],'
                                '"rekomendasi":[{"peran":"perencana_kota|insinyur_sipil|arsitek",'
                                '"tindakan":"teks","standar":"teks","dampak":"teks","prioritas":1}],'
                                '"prompt_gambar":"English prompt","before_after":"teks",'
                                '"caveat":"teks"}\n'
                                "Tanpa penjelasan, tanpa pagar markdown."),
                })
        del messages[base_len:]
    if last is not None:
        raise last
    hint = (" Balasan terpotong (finish_reason=length) - naikkan `max_tokens`."
            if truncated else "")
    raise HTTPException(
        status_code=502,
        detail=f"model tidak menghasilkan JSON valid; cuplikan balasan: {last_parse_error!r}.{hint}",
    )

@router.post("/advise", summary="Foto + metrik UVI -> rekomendasi perbaikan + prompt gambar")
async def llm_advise(
    file: UploadFile = File(..., description="Berkas gambar: jpg/jpeg/png/webp/gif/bmp"),
    metrics: Optional[str] = Form(None, description="JSON metrik segmentasi (opsional)"),
    scores: Optional[str] = Form(None, description="JSON 4 skor UVI (opsional)"),
    location: Optional[str] = Form(None, description="Nama/koordinat lokasi (opsional)"),
    context: Optional[str] = Form(None, description="Konteks tambahan (opsional)"),
    model: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    detail: str = Form("high"),
):
    """Analisa foto ruang publik + metriknya, balas rekomendasi terstruktur.

    Bedanya dengan `/llm/vision` (yang sekadar mendeskripsikan gambar): endpoint
    ini **wajib** membalas JSON berisi diagnosis, daftar masalah, rekomendasi
    untuk 3 peran (perencana kota / insinyur sipil / arsitek), plus `prompt_gambar`
    bahasa Inggris yang siap ditempel ke text-to-image (Midjourney/SDXL) untuk
    memvisualkan kondisi ideal.

    `metrics` & `scores` dikirim sebagai string JSON, mis:
        metrics={"vegetation_coverage_index":0.3477,"sky_openness_index":0.0223,...}
        scores={"beauty_score":4.2,"safety_score":3.1,"comfort_score":2.9,"uvi_score":0.36}
    Semua opsional; tanpa keduanya analisa dilakukan murni dari foto.
    """
    media_type = _vision_media_type(file.filename, file.content_type)
    image_bytes = await _read_vision_upload(file)
    if detail not in ("auto", "low", "high"):
        raise HTTPException(status_code=400, detail="parameter 'detail' harus auto|low|high")

    def _load_json(raw: Optional[str], name: str) -> Optional[dict]:
        if raw in (None, ""):
            return None
        try:
            val = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400,
                                detail=f"parameter '{name}' bukan JSON valid: {exc}")
        if not isinstance(val, dict):
            raise HTTPException(status_code=400, detail=f"parameter '{name}' harus objek JSON")
        return val

    metrics_obj = _load_json(metrics, "metrics")
    scores_obj = _load_json(scores, "scores")
    prompt = _advise_prompt(metrics=metrics_obj, scores=scores_obj,
                            location=location, extra=context)

    chosen = (model or "").strip() or settings.nine_router_advise_model
    out = await _ask_advise(
        model=chosen,
        prompt=prompt,
        image_bytes=image_bytes,
        media_type=media_type,
        max_tokens=_vision_int(max_tokens, name="max_tokens",
                               default=4096, lo=1, hi=ADVISE_MAX_TOKENS_CAP),
        temperature=_vision_float(temperature, name="temperature", default=None),
        detail=detail,
    )
    logger.info("POST /llm/advise model=%s %dms bytes=%d parse_ok=%s repair=%s",
                out["model_used"], out["elapsed_ms"], len(image_bytes),
                out["parse_ok"], out["repair_used"])
    raw = out["raw"]
    return {
        "id": raw.get("id") or f"chatcmpl-advise-{int(time.time() * 1000)}",
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
            "finish_reason": out.get("finish_reason") or "stop",
        }],
        "advice": out["advice"],
        "answer": out["content"],
        "context": {
            "upstream": _base_url(),
            "model": out["model_used"],
            "requested_model": out["requested_model"],
            "elapsed_ms": out["elapsed_ms"],
            "parse_ok": out["parse_ok"],
            "repair_used": out["repair_used"],
            "truncated": out.get("truncated", False),
            "image": {
                "filename": file.filename,
                "media_type": media_type,
                "bytes": len(image_bytes),
                "detail": detail,
            },
            "metrics_provided": sorted(metrics_obj.keys()) if metrics_obj else [],
            "scores_provided": sorted(scores_obj.keys()) if scores_obj else [],
            "location": location,
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
