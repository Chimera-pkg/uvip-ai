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
    POST /llm/generate             -> hasil /ai/process -> prompt gambar -> OpenAI
                                      Images API + saran. Balas JSON terstruktur.

Catatan penting: upstream SELALU membalas `text/event-stream` walaupun isinya
satu objek JSON utuh (bukan SSE sungguhan) — lihat `_extract_json()`. Tanpa
penanganan itu, `JSON.parse()` gagal dan semua permintaan tampak error.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from uvip_ai.config import settings
from uvip_ai.llm import infographic as infographic_mod
from uvip_ai.pipeline.evaluator import (
    compare_evaluations,
    evaluate_image,
    extract_before_metrics,
)
from uvip_ai.utils.storage import (
    GENERATED_DIR,
    INFOGRAPHICS_DIR,
    UPLOADS_DIR,
    output_path,
    rel_path,
    resolve_result_path,
    resolve_upload_path,
    upload_url,
)

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
                         timeout_s: Optional[float] = None,
                         url: Optional[str] = None,
                         key: Optional[str] = None) -> dict:
    """Panggil upstream /chat/completions (non-stream). Raise HTTPException bila gagal.

    ``timeout_s`` menimpa timeout global - dipakai /llm/advise supaya model yang
    menggantung cepat menyerah dan rantai berlanjut ke fallback.
    ``url``/``key`` menimpa endpoint utama - dipakai rantai fallback endpoint
    kedua (Cartethyia) untuk jalur teks.
    """
    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                  # WAJIB false: tanpa ini upstream membalas event-stream berisi
                  # JSON + "data: [DONE]" yang bikin parsing gagal.
                  "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if extra:
        body.update(extra)

    base = (url or _base_url()).rstrip("/")
    headers = {"Content-Type": "application/json"}
    auth_key = key if key is not None else settings.nine_router_key
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_s or settings.nine_router_timeout_s) as client:
            res = await client.post(f"{base}/chat/completions",
                                    headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502,
                            detail=f"upstream tidak terjangkau: {type(exc).__name__}: {exc}")

    data = _extract_json(res.text)
    elapsed_ms = int((time.time() - t0) * 1000)

    if res.status_code >= 400 or data.get("error"):
        err = data.get("error")
        # Upstream membalas DUA bentuk berbeda:
        #   {"error": "API key required for remote API access"}   <- STRING (401 gateway)
        #   {"error": {"message": "..."}}                          <- OBJEK (OpenAI-style)
        # Kode lama mengasumsikan selalu objek -> `.get()` pada str -> AttributeError
        # -> FastAPI membalas 500 "Internal Server Error" tanpa pesan yang berguna.
        if isinstance(err, dict):
            msg = err.get("message") or f"HTTP {res.status_code}"
        elif err:
            msg = str(err)
        else:
            msg = f"HTTP {res.status_code}"
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


def _fallback_endpoint() -> tuple[Optional[str], Optional[str]]:
    """(url, key) endpoint fallback kedua, atau (None, None) bila tak dikonfigurasi."""
    url = (settings.nine_router_fallback_url or "").strip()
    key = (settings.nine_router_fallback_key or "").strip()
    if not url:
        return None, None
    return url, key


async def _ask(*, model: str, messages: list, max_tokens: int,
               temperature: Optional[float]) -> dict:
    """Coba model utama, lalu fallback endpoint kedua (jalur TEKS).

    Model utama dicoba pada `nine_router_url`; bila gagal, `_fallbacks()`
    dicoba pada `nine_router_fallback_url` (bila dikonfigurasi). Jalur vision
    TIDAK memakai ini karena fallback teks tidak bisa membaca gambar.
    """
    chain = list(dict.fromkeys([model]))
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
    # Endpoint kedua (fallback) - khusus teks.
    fb_url, fb_key = _fallback_endpoint()
    for m in _fallbacks():
        if not fb_url:
            break
        try:
            out = await _call_upstream(model=m, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature,
                                       url=fb_url, key=fb_key)
            out["requested_model"] = model
            return out
        except HTTPException as exc:
            last = exc
            logger.warning("[llm] fallback model=%s gagal: %s", m, exc.detail)
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
    # membaca gambar. deepseek-flash adalah satu-satunya model "flash" yang
    # vision-capable, jadi rantai di sini TIDAK memakai _fallbacks() (yang
    # khusus teks dan diarahkan ke endpoint kedua).
    chain = list(dict.fromkeys([model]))
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

# 5 dimensi yang bisa dibobot user (slider prioritas, 0..1) untuk mengarahkan
# rekomendasi AI. Kunci pakai nama metrik; bobot tinggi = dimensi diprioritaskan.
# Ini satu-satunya sumber kebenaran untuk slider "prioritas" di demo.
ADVISE_WEIGHT_KEYS = (
    "building_visibility_index",   # Indeks visibilitas bangunan
    "vegetation_coverage_index",   # Indeks tutupan vegetasi
    "sky_openness_index",          # Indeks keterbukaan langit
    "building_coverage_pct",       # Indeks dominasi bangunan (building coverage)
    "ground_accessibility_index",  # Indeks aksesibilitas trotoar
)
ADVISE_WEIGHT_LABELS = {
    "building_visibility_index": "Visibilitas bangunan",
    "vegetation_coverage_index": "Tutupan vegetasi",
    "sky_openness_index": "Keterbukaan langit",
    "building_coverage_pct": "Dominasi bangunan",
    "ground_accessibility_index": "Aksesibilitas trotoar",
}

# System prompt domain. Ditulis eksplisit supaya output konsisten JSON dan
# rekomendasinya spesifik (mengutip angka), bukan saran generik "perbanyak pohon".
_ADVISE_SYSTEM = """Anda analis ruang publik perkotaan Indonesia (jalan, trotoar, alun-alun).
Anda menerima FOTO + metrik kuantitatif hasil segmentasi semantik (persentase kelas
dan indeks 0..1) + 4 skor persepsi (beauty/safety/comfort/uvi, skala 1..10 dan 0..1),
dan kadang BOBOT PRIORITAS (5 dimensi, 0..1) yang dipilih pengguna.

TUGAS: baca foto, kaitkan dengan angkanya, lalu beri rekomendasi perbaikan yang
SPESIFIK dan objektif untuk publik umum (bukan dibagi per profesi). Sudut pandang
satu: "apa yang perlu diperbaiki agar ruang ini lebih baik untuk semua orang".

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
6. Kalau BOBOT PRIORITAS diberikan: urutkan "rekomendasi" dari bobot tertinggi ke
   terendah, dan pertebal saran pada dimensi berbobot besar. Dimensi berbobot 0 boleh
   diabaikan. Sebutkan di "diagnosis" dimensi mana yang jadi prioritas.
7. "prompt_gambar" = prompt BAHASA INGGRIS untuk EDIT gambar (image-to-image), memuat:
   jenis ruang, material, vegetasi, pencahayaan. WAJIB menyertakan penegasan:
   posisi kamera, arah pandang, focal length, dan perspektif TIDAK berubah;
   struktur bangunan/pondasi dan tata letak TIDAK berubah; awan/langit/cuaca/
   pencahayaan/bayangan TIDAK ditambah atau diubah; elemen yang tidak disebut
   TIDAK ditambah/dihapus/dipindah; hanya [elemen yang diubah] yang diubah.
   Tulis kondisi IDEAL (perbaikan sudah dilakukan).
8. "caveat" wajib diisi: sebutkan bahwa analisa hanya dari 1 foto (tanpa data 3D,
   topografi, volume lalu lintas, atau survei lapangan) sehingga rekomendasi masih
   hipotesis yang perlu verifikasi lapangan.
9. Kalau metrik tidak diberikan, nilai murni dari foto dan tulis itu di "caveat".
10. RINGKAS - ini penting supaya balasan tidak terpotong:
   - "diagnosis": maksimal 3 kalimat.
   - "masalah": maksimal 4 item, tiap "temuan" maksimal 25 kata.
   - "rekomendasi": tepat 3 item objektif, tiap "tindakan" maksimal 40 kata.
   - "prompt_gambar": satu paragraf, maksimal 70 kata.
   - Jangan mengulang isi field di field lain.

SKEMA (semua field wajib ada; isi [] bila memang tidak ada):
{
  "diagnosis": "2-3 kalimat: kenapa kondisinya begini, kaitkan dengan angka",
  "masalah": [
    {"area": "trotoar", "temuan": "...", "bukti_angka": "walkability_ratio 0.024", "tingkat": "tinggi|sedang|rendah"}
  ],
  "rekomendasi": [
    {"peran": "umum", "tindakan": "...", "standar": "...", "dampak": "menaikkan safety_score", "prioritas": 1}
  ],
  "prompt_gambar": "English image-edit prompt keeping building structure and camera view unchanged",
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
    # Tiap rekomendasi WAJIB objek ber-`peran` (= "umum") - menjamin item
    # valid dan urut prioritas bisa dirender klien.
    for i, item in enumerate(recs):
        if not isinstance(item, dict):
            return False, f"rekomendasi[{i}] harus objek, bukan {type(item).__name__}"
        for k in ("peran", "tindakan"):
            if not str(item.get(k) or "").strip():
                return False, f"rekomendasi[{i}].{k} wajib diisi"
    return True, ""

def _advise_chain(model: str) -> list:
    """Rantai model untuk /llm/advise: model utama lalu `NINE_ROUTER_VISION_MODEL`.

    Fallback teks biasa belum tentu bisa membaca gambar. deepseek-flash adalah
    satu-satunya model "flash" vision-capable, jadi rantai ini TIDAK memakai
    `_fallbacks()` (khusus teks, endpoint kedua).
    """
    return list(dict.fromkeys([model, settings.nine_router_vision_model]))

def _advise_prompt(*, metrics: Optional[dict], scores: Optional[dict],
                   location: Optional[str], extra: Optional[str],
                   weights: Optional[dict] = None) -> str:
    """Susun pesan user: metrik + skor + lokasi sebagai konteks teks untuk LLM.

    Angka inilah yang membuat rekomendasi spesifik ("trotoar hanya 2,4% dari
    lebar jalan") alih-alih generik ("perbanyak trotoar"). ``weights`` (opsional)
    = bobot prioritas 5 dimensi (0..1) yang dipilih pengguna; LLM mengurutkan
    rekomendasinya mengikuti bobot.
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
    if weights:
        lines.append("\nBOBOT PRIORITAS (0..1, dari pengguna) - urutkan rekomendasi "
                     "dari bobot tertinggi:")
        for k in ADVISE_WEIGHT_KEYS:
            if k in weights:
                label = ADVISE_WEIGHT_LABELS.get(k, k)
                try:
                    w = float(weights[k])
                except (TypeError, ValueError):
                    w = 0.0
                lines.append(f"  - {label} ({k}): {w:.3f}")
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
                                '"rekomendasi":[{"peran":"umum",'
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

def result_context(result: dict) -> tuple[dict, dict]:
    """Ambil (metrics, scores) dari keluaran `/ai/process`.

    Dipakai `/ai/advise` supaya klien tidak perlu menyalin angka dari respons
    segmentasi ke parameter prompt: metrik = gabungan persentase kelas dan
    indeks (`_seg_view`), skor = isi `perception_prediction`.
    """
    if not isinstance(result, dict):
        return {}, {}
    metrics = _seg_view(result)
    perc = result.get("perception_prediction")
    scores = ({k: v for k, v in perc.items() if isinstance(v, (int, float))}
              if isinstance(perc, dict) else {})
    return metrics, scores


async def advise_photo(*, image_bytes: bytes, filename: Optional[str] = None,
                       media_type: Optional[str] = None,
                       metrics: Optional[dict] = None, scores: Optional[dict] = None,
                       weights: Optional[dict] = None,
                       location: Optional[str] = None, context: Optional[str] = None,
                       model: Optional[str] = None, max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       detail: str = "high") -> dict:
    """Inti `/llm/advise` tanpa HTTP — dipakai juga oleh `/ai/advise`.

    Melempar `HTTPException` yang sama seperti endpoint HTTP (400/502) supaya
    pemanggil non-HTTP pun mendapat pesan error yang konsisten.
    """
    if detail not in ("auto", "low", "high"):
        raise HTTPException(status_code=400, detail="parameter 'detail' harus auto|low|high")
    media_type = media_type or _vision_media_type(filename, None)
    prompt = _advise_prompt(metrics=metrics, scores=scores,
                            location=location, extra=context, weights=weights)
    out = await _ask_advise(
        model=(model or "").strip() or settings.nine_router_advise_model,
        prompt=prompt,
        image_bytes=image_bytes,
        media_type=media_type,
        max_tokens=max_tokens if max_tokens is not None else 4096,
        temperature=temperature,
        detail=detail,
    )
    out["media_type"] = media_type
    out["prompt"] = prompt
    out["context"] = {
        "upstream": _base_url(),
        "model": out["model_used"],
        "requested_model": out["requested_model"],
        "elapsed_ms": out["elapsed_ms"],
        "parse_ok": out["parse_ok"],
        "repair_used": out["repair_used"],
        "truncated": out.get("truncated", False),
        "image": {
            "filename": filename,
            "media_type": media_type,
            "bytes": len(image_bytes),
            "detail": detail,
        },
        "metrics_provided": sorted(metrics) if metrics else [],
        "scores_provided": sorted(scores) if scores else [],
        "weights_provided": sorted(weights) if weights else [],
        "location": location,
    }
    return out


@router.post("/advise", summary="Foto + metrik UVI -> rekomendasi perbaikan + prompt gambar")
async def llm_advise(
    file: UploadFile = File(..., description="Berkas gambar: jpg/jpeg/png/webp/gif/bmp"),
    metrics: Optional[str] = Form(None, description="JSON metrik segmentasi (opsional)"),
    scores: Optional[str] = Form(None, description="JSON 4 skor UVI (opsional)"),
    weights: Optional[str] = Form(None, description="JSON 5 bobot prioritas 0..1 (opsional)"),
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
    objektif (persona "umum"), plus `prompt_gambar` bahasa Inggris untuk EDIT
    gambar (image-to-image) yang memvisualkan kondisi ideal tanpa mengubah
    struktur bangunan dan sudut kamera.

    `metrics`, `scores` & `weights` dikirim sebagai string JSON, mis:
        metrics={"vegetation_coverage_index":0.3477,"sky_openness_index":0.0223,...}
        scores={"beauty_score":4.2,"safety_score":3.1,"comfort_score":2.9,"uvi_score":0.36}
        weights={"vegetation_coverage_index":0.5,"sky_openness_index":0.2,...}
    Semua opsional; tanpa ketiganya analisa dilakukan murni dari foto.
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

    weights_raw = _load_json(weights, "weights")
    if weights_raw:
        # Terima hanya 5 dimensi yang dikenal; buang yang tak dikenal supaya
        # LLM tidak menerima key asing, dan clamp 0..1.
        clean_weights: dict = {}
        for k in ADVISE_WEIGHT_KEYS:
            v = weights_raw.get(k)
            if v is None:
                continue
            try:
                clean_weights[k] = min(max(float(v), 0.0), 1.0)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400,
                                    detail=f"weights['{k}'] harus angka 0..1")
        if clean_weights:
            weights_raw = clean_weights
        else:
            weights_raw = None

    out = await advise_photo(
        image_bytes=image_bytes,
        filename=file.filename,
        media_type=media_type,
        metrics=_load_json(metrics, "metrics"),
        scores=_load_json(scores, "scores"),
        weights=weights_raw,
        location=location,
        context=context,
        model=model,
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
        "context": out["context"],
        "usage": raw.get("usage"),
    }

# ─── POST /llm/generate: baca hasil /ai/process -> generate ulang foto ───

# Ukuran valid gpt-image-1.5 (resmi OpenAI).
_OPENAI_IMAGE_SIZES = ("1024x1024", "1536x1024", "1024x1536")


def _num(d: Optional[dict], key: str) -> Optional[float]:
    """Ambil angka dari dict, toleran None/str. Kembalikan None kalau tidak ada."""
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct_or_index(d: Optional[dict], *keys: str) -> Optional[float]:
    """Ambil angka pertama dari daftar kunci yang ada. Nilai tetap (persen/index)."""
    for k in keys:
        val = _num(d, k)
        if val is not None:
            return val
    return None


def _seg_view(target: dict) -> dict:
    """Gabungan angka segmentasi dari dua tempat di keluaran /ai/process.

    ``metrics_source.pct_by_class`` memakai nama kelas mentah
    (``vegetation``, ``sidewalk``, ``road``, ``vehicle`` ...), sedangkan
    ``segmentation_results`` memakai nama metrik (``green_coverage_pct``,
    ``walkability_ratio`` ...). Keduanya digabung; yang terakhir menang.
    """
    out: dict = {}
    src = target.get("metrics_source") if isinstance(target, dict) else None
    if isinstance(src, dict):
        pbc = src.get("pct_by_class")
        if isinstance(pbc, dict):
            out.update({k: v for k, v in pbc.items() if isinstance(v, (int, float))})
    seg = target.get("segmentation_results") if isinstance(target, dict) else None
    if isinstance(seg, dict):
        out.update({k: v for k, v in seg.items() if isinstance(v, (int, float))})
    return out


def _veh_pct(seg: dict) -> Optional[float]:
    """Nilai dominasi kendaraan dalam persen (0..100)."""
    v = _pct_or_index(seg, "vehicle_pct", "vehicle")
    if v is None:
        return None
    # Kalau yang terdeteksi hanya index excel_mode (0..1), ubah ke persen.
    if "vehicle_pct" not in seg and "vehicle" not in seg and v <= 1.0:
        return v * 100.0
    return v


def _simple_image_prompt(target: dict) -> str:
    """Prompt gambar SESEMPEL MUNGKIN dari angka target UVI/PVI/persen.

    Satu baris bahasa Inggris: hanya menyebut elemen yang perlu diubah.
    """
    seg = _seg_view(target)

    veg = _pct_or_index(seg, "vegetation_pct", "vegetation", "green_coverage_pct") or 0.0
    sky = _pct_or_index(seg, "sky_pct", "sky", "sky_visibility_pct") or 0.0
    sw = _pct_or_index(seg, "sidewalk_pct", "sidewalk") or 0.0
    veh = _veh_pct(seg) or 0.0
    bld = _pct_or_index(seg, "building_pct", "building", "building_coverage_pct") or 0.0

    ph = ["Redesign this street"]
    if veg < 20:
        ph.append("add more trees and greenery")
    if sky < 25:
        ph.append("open up the sky view")
    if sw < 12:
        ph.append("widen the pedestrian sidewalk")
    if veh > 15:
        ph.append("reduce vehicle dominance")
    if bld > 40:
        ph.append("provide more building setbacks")
    if len(ph) == 1:  # pastikan selalu ada aksi
        ph.append("improve the overall street environment")
    ph.append("keep the exact camera position, viewing direction, focal length, "
              "and perspective; keep all building structures, foundations, and "
              "layout unchanged; do not add, remove, or change clouds, sky, "
              "weather, lighting, or shadows; do not add, remove, relocate, or "
              "alter any element not explicitly mentioned; only modify the "
              "mentioned elements; photorealistic")
    return ", ".join(ph) + "."


def _simple_suggestions(target: dict) -> list:
    """Saran = kalimat bebas (tanpa prioritas), diturunkan dari angka target.

    Bukan LLM: murni template dari angka — deterministik, biaya nol, dan tidak
    mungkin mengarang angka. Threshold diambil dari standar praktis.
    """
    seg = _seg_view(target)
    perc = target.get("perception_prediction") if isinstance(target, dict) else None
    perc = perc if isinstance(perc, dict) else {}

    veg = _pct_or_index(seg, "vegetation_pct", "vegetation", "green_coverage_pct")
    sky = _pct_or_index(seg, "sky_pct", "sky", "sky_visibility_pct")
    sw = _pct_or_index(seg, "sidewalk_pct", "sidewalk")
    veh = _veh_pct(seg)
    bld = _pct_or_index(seg, "building_pct", "building", "building_coverage_pct")
    walk = _num(seg, "walkability_ratio")
    uvi = _num(perc, "uvi_score")

    sents = []
    if veg is not None and veg < 25:
        sents.append(
            f"Tingkatkan tutupan vegetasi dari {veg:.1f}% dengan menambah "
            "pohon peneduh dan tanaman perdu di sepanjang koridor serta selasar bangunan.")
    if sky is not None and sky < 25:
        sents.append(
            f"Buka pandangan langit dari {sky:.1f}% dengan merapikan kanopi pohon "
            "dan meniadakan papan iklan yang menghalangi.")
    if sw is not None and sw < 12:
        sents.append(
            f"Perlebar dan sambungkan trotoar (saat ini {sw:.1f}% dari komposisi) "
            "agar aman dan nyaman bagi pejalan kaki.")
    if walk is not None and walk < 0.15:
        sents.append(
            f"Rasio pejalan kaki terhadap badan jalan hanya {walk:.2f} "
            "(ideal > 0.25). Alihkan sebagian ruang jalan menjadi trotoar "
            "yang menerus dan bebas hambatan.")
    if veh is not None and veh > 15:
        sents.append(
            f"Kurangi dominasi kendaraan ({veh:.1f}%) dengan manajemen parkir "
            "dan pembatasan lalu lintas di kawasan pejalan.")
    if bld is not None and bld > 40:
        sents.append(
            f"Atur jarak antar bangunan ({bld:.1f}%) agar sirkulasi udara dan "
            "jarak pandang lebih terbuka.")
    if uvi is not None and uvi < 0.7:
        sents.append(
            f"Nilai UVI saat ini {uvi:.2f} (skala 0..1) — rendah. Terapkan "
            "langkah di atas secara bersama untuk menaikkannya.")
    if not sents:
        sents.append(
            "Lakukan verifikasi lapangan dan bandingkan dengan foto sebelum menentukan "
            "prioritas perbaikan.")
    return sents[:4]





_MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
                   ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Khusus EDIT gambar: hanya format foto, bukan video.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _raw_image_url(photo_path: Optional[str]) -> Optional[str]:
    """URL publik foto mentah; None kalau file tidak ada / di luar uploads/."""
    if not photo_path:
        return None
    try:
        path = resolve_upload_path(photo_path)
    except (ValueError, PermissionError):
        return None
    if path.suffix.lower() not in _MEDIA_SUFFIXES:
        return None
    return upload_url(path) if path.is_file() else None


def _raw_image_path(photo_path: Optional[str]) -> Optional[Path]:
    """Path file raw (untuk EDIT image) atau None bila tak tersedia/harusnya video."""
    if not photo_path:
        return None
    try:
        path = resolve_upload_path(photo_path)
    except (ValueError, PermissionError):
        return None
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    return path if path.is_file() else None


def _load_target_json(result: str) -> dict:
    """Terima isi JSON hasil /ai/process ATAU referensi ke file ``result_*.json``.

    Referensi boleh berupa path/URL file result, path/URL artefak lain
    (``overlay_*.jpg``, ``mask_*.jpg``), semua di dalam ``uploads/``.
    """
    strip = (result or "").strip()
    if not strip:
        raise HTTPException(status_code=400, detail="param 'result' wajib diisi")
    if strip.startswith("{"):
        try:
            val = json.loads(strip)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400,
                                detail=f"'result' bukan JSON valid: {exc}")
    else:
        try:
            path, _origin = resolve_result_path(strip)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            val = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400,
                                detail=f"file result bukan JSON valid: {exc}")
    if not isinstance(val, dict):
        raise HTTPException(status_code=400, detail="'result' harus objek JSON")
    return val


def _save_generated(b64_json: str, stem: str) -> Path:
    """Simpan PNG hasil OpenAI dengan nama ``img_<stem>_<epoch_ms>.png``."""
    path = output_path(GENERATED_DIR, "img_", stem, ".png")
    path.write_bytes(base64.b64decode(b64_json))
    return path


def _target_stem(target: dict, photo_path: Optional[str]) -> str:
    """Ambil stem asli dari nama artefak (buang prefix + timestamp)."""
    for value in (target.get("result_path"), target.get("result_url"), photo_path):
        if not isinstance(value, str) or not value.strip():
            continue
        stem = Path(value).stem
        m = re.match(
            r"^(?:result_video|result|overlay|seg|mask|img|video)_"
            r"(.+?)_(?:\d{8}_\d{6}|\d{10,13})(?:_\d+)?$",
            stem,
        )
        if m:
            return m.group(1)[:120]
        if stem and not re.search(r"_(?:\d{8}_\d{6}|\d{10,13})(?:_\d+)?$", stem):
            return stem[:120]
    return "gen"


async def _openai_generate_image(*, prompt: str, model: str, size: str,
                                 count: int) -> dict:
    """Panggil OpenAI Images API (gpt-image-1.5), minta base64.

    Kembalikan {"items": [{"b64_json":..., "revised_prompt":...}], ...}.
    """
    key = settings.openai_api_key.strip()
    if not key:
        raise HTTPException(status_code=503,
                            detail="OPENAI_API_KEY kosong - set di .env lalu restart")
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # CATATAN: gpt-image-1.5 SELALU mengembalikan b64_json dan MENOLAK parameter
    # `response_format` (400 unknown_parameter). Jangan tambahkan lagi.
    payload = {"model": model, "prompt": prompt, "n": count, "size": size}
    try:
        async with httpx.AsyncClient(timeout=settings.openai_image_timeout_s) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502,
                            detail=f"gagal hubungi OpenAI Images API: {exc}")
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code,
                            detail=f"OpenAI Images API {resp.status_code}: {detail}")
    data = resp.json()
    items = data.get("data") or []
    if not items:
        raise HTTPException(status_code=502,
                            detail="OpenAI Images API balasan tanpa data")
    out = []
    for it in items:
        b64 = it.get("b64_json")
        if not b64:
            raise HTTPException(status_code=502,
                                detail="OpenAI Images API balasan tanpa b64_json")
        out.append({"b64_json": b64, "revised_prompt": it.get("revised_prompt", prompt)})
    return {"items": out, "model": data.get("model", model)}


async def _openai_edit_image(*, image_path: Path, prompt: str, model: str,
                             size: str, count: int) -> dict:
    """EDIT gambar (image-to-image) pakai OpenAI ``/v1/images/edits``.

    Kirim foto RAW sebagai input sehingga struktur bangunan, pondasi, tata
    letak, dan sudut kamera dipertahankan — hanya elemen di prompt yang diubah.
    Bentuk balasan sama dengan ``_openai_generate_image``.
    """
    key = settings.openai_api_key.strip()
    if not key:
        raise HTTPException(status_code=503,
                            detail="OPENAI_API_KEY kosong - set di .env lalu restart")
    url = "https://api.openai.com/v1/images/edits"
    media_type = _vision_media_type(image_path.name, None)
    data = {"model": model, "prompt": prompt, "n": str(count), "size": size}
    files = {"image": (image_path.name, image_path.read_bytes(), media_type)}
    try:
        async with httpx.AsyncClient(timeout=settings.openai_image_timeout_s) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                data=data,
                files=files,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502,
                            detail=f"gagal hubungi OpenAI Images edits API: {exc}")
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code,
                            detail=f"OpenAI Images edits API {resp.status_code}: {detail}")
    payload = resp.json()
    items = payload.get("data") or []
    if not items:
        raise HTTPException(status_code=502,
                            detail="OpenAI Images edits API balasan tanpa data")
    out = []
    for it in items:
        b64 = it.get("b64_json")
        if not b64:
            raise HTTPException(status_code=502,
                                detail="OpenAI Images edits API balasan tanpa b64_json")
        out.append({"b64_json": b64, "revised_prompt": it.get("revised_prompt", prompt)})
    return {"items": out, "model": payload.get("model", model)}





@router.post("/generate", summary="Baca hasil /ai/process -> generate ulang foto (OpenAI Images) + saran + evaluasi 8 indeks")
async def llm_generate(
    result: str = Form(..., description="JSON hasil /ai/process, atau path URL file result_*.json di server"),
    photo_path: Optional[str] = Form(None, description="Path gambar raw di server (opsional, untuk raw_image_url)"),
    prompt: Optional[str] = Form(None, description="Override prompt gambar (opsional)"),
    model: Optional[str] = Form(None, description="Model OpenAI images (default gpt-image-1.5)"),
    size: Optional[str] = Form(None, description="1024x1024 | 1536x1024 | 1024x1536"),
    count: Optional[str] = Form(None, description="Jumlah gambar, 1..4 (default 1)"),
    evaluate: bool = Form(True, description="Otomatis analisis segmentasi 8 indeks pada gambar hasil AI"),
):
    """Alur satu endpoint "AI photo generator":

    1. ``result`` = keluaran POST /ai/process (JSON inline) atau referensi
       path/URL ``result_*.json`` di server -> dibaca variabel UVI/PVI &
       persentase (``perception_prediction``, ``segmentation_results``).
    2. prompt gambar disusun SESEMPEL MUNGKIN dari angka target, dan memuat
       penegasan "struktur bangunan & sudut kamera tidak berubah".
    3. Bila foto RAW tersedia -> EDIT image (``/v1/images/edits``) supaya
       pondasi/struktur/POV benar-benar dipertahankan; tanpa RAW -> fallback
       text-to-image (``/v1/images/generations``).
    4. saran = kalimat bebas (tanpa prioritas) dari angka, bukan LLM => gratis &
       deterministik.
    5. Response berisi ``generated_image_url`` + ``raw_image_url`` + suggestions
       supaya bisa ditampilkan berdampingan (raw | saran | hasil AI).
    6. Bila ``evaluate=True``, gambar hasil AI otomatis disegmentasi dengan SegFormer-B5
       untuk menghasilkan 8 indeks komposit, skor UVI baru, overlay visual, dan
       perbandingan Before vs After.
    """
    target = _load_target_json(result)

    chosen_model = (model or "").strip() or settings.openai_image_model
    chosen_size = (size or "").strip() or settings.openai_image_size
    if chosen_size not in _OPENAI_IMAGE_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"size harus salah satu: {', '.join(_OPENAI_IMAGE_SIZES)}",
        )
    try:
        count_n = int(count) if count else 1
    except ValueError:
        raise HTTPException(status_code=400, detail="count harus bilangan bulat")
    if not 1 <= count_n <= 4:
        raise HTTPException(status_code=400, detail="count harus 1..4")

    image_prompt = (prompt or "").strip() or _simple_image_prompt(target)

    # Foto RAW untuk EDIT image: prioritas `photo_path`, lalu URL artefak asli
    # dari result (original > segmentation > mask). Tanpa RAW -> generations.
    raw_path = (_raw_image_path(photo_path)
                or _raw_image_path(target.get("original_image_url"))
                or _raw_image_path(target.get("segmentation_url")))
    t0 = time.time()
    if raw_path is not None:
        gen = await _openai_edit_image(image_path=raw_path, prompt=image_prompt,
                                       model=chosen_model, size=chosen_size,
                                       count=count_n)
        mode = "edit"
    else:
        gen = await _openai_generate_image(prompt=image_prompt, model=chosen_model,
                                           size=chosen_size, count=count_n)
        mode = "generate"
    stem = _target_stem(target, photo_path)
    paths = [_save_generated(item["b64_json"], stem) for item in gen["items"]]
    saved = [u for u in (upload_url(p) for p in paths) if u]
    elapsed_ms = int((time.time() - t0) * 1000)

    raw_image_url = (_raw_image_url(photo_path)
                     or _raw_image_url(target.get("original_image_url"))
                     or _raw_image_url(target.get("segmentation_url")))

    eval_block = None
    generated_metrics = None
    generated_scores = None
    generated_overlay_url = None

    if evaluate and paths:
        try:
            target_path = paths[0]
            eval_res = evaluate_image(target_path, stem=stem, save_overlay=True)
            before_indices, before_scores = extract_before_metrics(target)
            comp = compare_evaluations(
                before_indices, eval_res["indices"],
                before_scores, eval_res["scores"]
            )
            generated_metrics = eval_res["indices"]
            generated_scores = eval_res["scores"]
            generated_overlay_url = eval_res["overlay_url"]
            eval_block = {
                "before": {
                    "indices": before_indices,
                    "scores": before_scores,
                },
                "after": {
                    "indices": eval_res["indices"],
                    "scores": eval_res["scores"],
                    "segmentation_results": eval_res["segmentation_results"],
                    "overlay_url": eval_res["overlay_url"],
                    "segmentation_url": eval_res.get("segmentation_url"),
                },
                "delta": comp["indices"],
                "scores_delta": comp["scores"],
                "summary": comp["summary"],
                "elapsed_ms": eval_res.get("elapsed_ms", 0),
            }
        except Exception as exc:
            logger.warning("Gagal auto-evaluasi segmentasi gambar AI: %s", exc)

    return {
        "generated_image_url": saved[0] if saved else None,
        "generated_image_urls": saved,
        "generated_paths": [rel_path(p) for p in paths],
        "raw_image_url": raw_image_url,
        "photo_path": photo_path,
        "target": {
            "perception_prediction": target.get("perception_prediction"),
            "segmentation_results": target.get("segmentation_results"),
        },
        "suggestions": _simple_suggestions(target),
        "prompt_image": image_prompt,
        "revised_prompt": gen["items"][0]["revised_prompt"],
        "image_mode": mode,
        "image_model": chosen_model,
        "size": chosen_size,
        "count": count_n,
        "elapsed_ms": elapsed_ms,
        "generated_metrics": generated_metrics,
        "generated_scores": generated_scores,
        "generated_overlay_url": generated_overlay_url,
        "evaluation": eval_block,
        "usage": None,
    }


@router.post("/evaluate-generated", summary="Analisis segmentasi 8 indeks pada gambar hasil AI yang sudah ada")
async def llm_evaluate_generated(
    generated_path: str = Form(..., description="Path gambar AI atau URL di uploads/generated/"),
    result: Optional[str] = Form(None, description="Path result_*.json atau JSON hasil /ai/process foto asli (opsional)"),
):
    """Jalankan analisis segmentasi 8 indeks pada gambar AI yang sudah tersimpan di server.

    Jika `result` disertakan, respons akan memuat perbandingan Before vs After (delta).
    """
    try:
        resolved_path = resolve_upload_path(generated_path)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not resolved_path.is_file():
        raise HTTPException(status_code=404, detail=f"File gambar tidak ditemukan: {generated_path}")

    stem = _target_stem({"result_path": str(resolved_path)}, str(resolved_path))
    try:
        eval_res = evaluate_image(resolved_path, stem=stem, save_overlay=True)
    except Exception as exc:
        logger.error("Gagal evaluasi segmentasi: %s", exc)
        raise HTTPException(status_code=500, detail=f"Gagal evaluasi segmentasi: {exc}")

    comparison = None
    before_block = None
    if result:
        try:
            target = _load_target_json(result)
            before_indices, before_scores = extract_before_metrics(target)
            comp = compare_evaluations(
                before_indices, eval_res["indices"],
                before_scores, eval_res["scores"]
            )
            before_block = {
                "indices": before_indices,
                "scores": before_scores,
            }
            comparison = {
                "delta": comp["indices"],
                "scores_delta": comp["scores"],
                "summary": comp["summary"],
            }
        except Exception as exc:
            logger.warning("Gagal memproses perbandingan dengan result: %s", exc)

    return {
        "generated_image_url": upload_url(resolved_path),
        "generated_overlay_url": eval_res.get("overlay_url"),
        "generated_metrics": eval_res.get("indices"),
        "generated_scores": eval_res.get("scores"),
        "segmentation_results": eval_res.get("segmentation_results"),
        "before": before_block,
        "comparison": comparison,
        "elapsed_ms": eval_res.get("elapsed_ms", 0),
    }


async def _ask_infographic(*, model: str, prompt: str, before_bytes: Optional[bytes],
                           before_type: Optional[str], after_bytes: bytes,
                           after_type: str, max_tokens: int) -> dict:
    """LLM vision (before + after, atau after saja) -> JSON infografis terstruktur."""
    b64_after = base64.b64encode(after_bytes).decode("ascii")
    content: list = [{"type": "text", "text": prompt}]
    if before_bytes:
        b64_before = base64.b64encode(before_bytes).decode("ascii")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{before_type};base64,{b64_before}", "detail": "high"}})
    content.append({"type": "image_url",
                    "image_url": {"url": f"data:{after_type};base64,{b64_after}", "detail": "high"}})
    messages: list = [
        {"role": "system", "content": infographic_mod.INFOGRAPHIC_SYSTEM},
        {"role": "user", "content": content},
    ]
    last: Optional[HTTPException] = None
    last_parse_error = ""
    for m in _advise_chain(model):
        base_len = len(messages)
        for attempt in (1, 2):
            try:
                out = await _call_upstream(model=m, messages=messages,
                                           max_tokens=max_tokens, temperature=None,
                                           timeout_s=settings.nine_router_timeout_s)
            except HTTPException as exc:
                last = exc
                logger.warning("[llm] infographic model=%s gagal: %s", m, exc.detail)
                break
            finish = out.get("finish_reason") or ""
            parsed = _extract_json_object(out["content"])
            ok, why = (False, "tidak ada objek JSON")
            if parsed is not None:
                ok, why = infographic_mod.infographic_schema_ok(parsed)
            if ok and finish != "length":
                out["infographic"] = parsed
                out["parse_ok"] = True
                return out
            last_parse_error = out["content"][:300]
            logger.warning("[llm] infographic model=%s gagal validasi (attempt %d): %s",
                           m, attempt, why)
            if attempt == 1:
                messages.append({"role": "assistant", "content": out["content"][:2000]})
                messages.append({"role": "user", "content": (
                    f"Balasan tadi tidak sesuai skema ({why}). Kirim ULANG HANYA "
                    "satu objek JSON dengan bentuk persis di system prompt. "
                    "Tanpa penjelasan, tanpa pagar markdown.")})
        del messages[base_len:]
    if last is not None:
        raise last
    raise HTTPException(status_code=502,
                        detail=f"model tidak menghasilkan JSON valid: {last_parse_error!r}")


@router.post("/infographic", summary="Foto hasil generate AI -> analisis 10 intervensi + infografis PNG")
async def llm_infographic(
    generated_path: str = Form(..., description="Path/URL gambar hasil AI di uploads/generated/"),
    result: Optional[str] = Form(None, description="result_*.json atau JSON hasil /ai/process (opsional, untuk foto asli + UVI before)"),
    photo_path: Optional[str] = Form(None, description="Path foto asli (BEFORE) bila result kosong"),
    project_name: Optional[str] = Form(None, description="Nama proyek (opsional)"),
    photo_id: Optional[str] = Form(None, description="Photo ID (opsional)"),
    location: Optional[str] = Form(None, description="Nama lokasi (opsional)"),
    city: Optional[str] = Form(None, description="Kota/Region (opsional)"),
    urban_character: Optional[str] = Form(None, description="Karakter urban (opsional, else infer LLM)"),
    heritage_status: Optional[str] = Form(None, description="Status heritage (opsional)"),
):
    """Langkah 3 rangkaian: terima foto hasil generate -> infografis (TIDAK generate ulang).

    1. Resolve ``generated_path`` (AFTER).
    2. Load ``result`` (opsional) -> BEFORE (foto asli) + UVI + 5 indikator.
    3. LLM vision 2 gambar (before + after) -> 10 intervensi + regulasi + validasi.
    4. Render infografis 2-kolom programatik (Pillow) -> PNG.
    """
    # 1. AFTER = foto hasil generate AI.
    try:
        after_path = resolve_upload_path(generated_path)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not after_path.is_file():
        raise HTTPException(status_code=404, detail=f"gambar hasil AI tidak ditemukan: {generated_path}")

    # 2. BEFORE + UVI + 5 indikator dari result (opsional).
    seg: dict = {}
    perc: dict = {}
    cap: dict = {}
    lat: Optional[float] = None
    lon: Optional[float] = None
    target: Optional[dict] = None
    if result and result.strip():
        target = _load_target_json(result)
        seg = target.get("segmentation_results") or {}
        perc = target.get("perception_prediction") or {}
        cap = target.get("capture_location") or {}
        if isinstance(cap, dict):
            lat = cap.get("latitude")
            lon = cap.get("longitude")
    before_path: Optional[Path] = None
    if target is not None:
        before_path = (_raw_image_path(target.get("original_image_url"))
                       or _raw_image_path(target.get("segmentation_url")))
    before_path = before_path or _raw_image_path(photo_path)

    # Fallback: turunkan BEFORE + UVI dari nama file gambar hasil generate
    # (img_<stem>_<epoch>.png -> uploads/originals/<stem>.<ext> + result_<stem>_*.json).
    stem = _target_stem({"result_path": str(after_path)}, str(after_path))
    if before_path is None:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = UPLOADS_DIR / "originals" / f"{stem}{ext}"
            if candidate.is_file():
                before_path = candidate
                break
    if not seg and not perc:
        result_files = sorted((UPLOADS_DIR / "results").glob(f"result_{stem}_*.json"))
        if result_files:
            try:
                tgt = json.loads(result_files[-1].read_text(encoding="utf-8"))
                if isinstance(tgt, dict):
                    seg = tgt.get("segmentation_results") or {}
                    perc = tgt.get("perception_prediction") or {}
                    cap = tgt.get("capture_location") or {}
                    if isinstance(cap, dict):
                        lat = cap.get("latitude")
                        lon = cap.get("longitude")
                    if before_path is None:
                        before_path = (_raw_image_path(tgt.get("original_image_url"))
                                       or _raw_image_path(tgt.get("segmentation_url")))
            except (OSError, json.JSONDecodeError):
                pass

    # 3. Lokasi fallback: EXIF/result > foto asli > None.
    loc = (location or "").strip()
    if not loc and isinstance(cap, dict) and cap.get("latitude") is not None:
        loc = f"{float(cap['latitude']):.5f}, {float(cap['longitude']):.5f}"
    city_val = (city or "").strip() or None

    prompt_text = infographic_mod.build_infographic_prompt(
        metrics=seg, scores=perc,
        project_name=project_name, photo_id=photo_id, location=loc, city=city_val,
        lat=lat, lon=lon, urban_character=urban_character, heritage_status=heritage_status,
    )
    try:
        out = await _ask_infographic(
            model=settings.nine_router_advise_model,
            prompt=prompt_text,
            before_bytes=before_path.read_bytes() if before_path else None,
            before_type=_vision_media_type(before_path.name, None) if before_path else None,
            after_bytes=after_path.read_bytes(),
            after_type=_vision_media_type(after_path.name, None),
            max_tokens=8192,
        )
    except HTTPException as exc:
        # LLM gagal -> tetap kembalikan foto + prompt (tanpa analisis).
        return {
            "generated_image_url": upload_url(after_path),
            "infographic": None,
            "infographic_url": None,
            "error": str(exc.detail),
            "prompt": prompt_text,
        }

    advice = out["infographic"]

    # Infografis AI: gpt-image-1.5 edit dari foto AFTER + prompt poster pendek.
    stem = _target_stem({"result_path": str(after_path)}, str(after_path))
    uvi = perc.get("uvi_score")
    uvi_val = float(uvi) if isinstance(uvi, (int, float)) else None
    image_prompt = infographic_mod.build_infographic_image_prompt(
        advice=advice, metrics=seg, uvi_score=uvi_val,
        project_name=project_name, photo_id=photo_id, location=loc,
        city=city_val, lat=lat, lon=lon,
        urban_character=urban_character, heritage_status=heritage_status,
    )
    info_path = None
    try:
        gen_info = await _openai_edit_image(
            image_path=after_path, prompt=image_prompt,
            model=settings.openai_image_model, size="1536x1024", count=1,
        )
        info_path = output_path(INFOGRAPHICS_DIR, "infographic_", stem, ".png")
        info_path.write_bytes(base64.b64decode(gen_info["items"][0]["b64_json"]))
    except HTTPException as exc:
        # Fallback: render Pillow supaya tetap ada PNG walau AI gagal.
        logger.warning("Gagal generate infografis AI (%s) - fallback Pillow", exc.detail)
        meta = {
            "project_name": (project_name or "").strip(),
            "photo_id": (photo_id or "").strip(),
            "location": loc,
            "city": city_val or "",
            "urban_character": str(advice.get("urban_character") or ""),
            "heritage_status": str(advice.get("heritage_status") or ""),
        }
        info_path = output_path(INFOGRAPHICS_DIR, "infographic_", stem, ".png")
        infographic_mod.render_infographic_png(
            photo_path=after_path, advice=advice, meta=meta,
            uvi_score=uvi_val, out_path=info_path,
        )

    return {
        "generated_image_url": upload_url(after_path),
        "infographic_url": upload_url(info_path) if info_path else None,
        "infographic": advice,
        "raw_image_url": upload_url(before_path),
        "model": out.get("model_used"),
        "llm_elapsed_ms": out.get("elapsed_ms"),
        "parse_ok": out.get("parse_ok", False),
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
