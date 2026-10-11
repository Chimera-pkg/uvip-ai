/**
 * 9router GET-shim (Bun) — LAMA
 * ------------------------------------------------------------------
 * DIGANTIKAN oleh router Python: `src/uvip_ai/llm/router.py` (prefix /llm/*),
 * yang menempel di API utama sehingga muncul di /docs dan tidak butuh proses
 * terpisah. File ini disimpan hanya sebagai referensi.
 *
 * PERUBAHAN: API key TIDAK lagi di-hardcode di sini. Ambil dari environment:
 *     NINE_ROUTER_KEY=...  bun server.mjs
 * Key yang sama sudah dipindah ke .env (gitignored) untuk router Python.
 *
 * Upstream : https://9router-production-fc96.up.railway.app/v1
 *           (OpenAI-compatible, tapi HANYA POST -> GET balas 405)
 *
 * Shim ini menambah endpoint GET supaya prompt bisa dipanggil
 * langsung dari browser / <iframe> / <img> / curl / fetch.
 *
 *   GET  /ask?q=...&model=...&system=...     -> text/plain  (jawaban mentah)
 *   GET  /v1/chat/completions?q=...&model=.. -> JSON OpenAI-shape (+ context)
 *   GET  /v1/models                          -> JSON daftar model
 *   POST /v1/chat/completions                -> passthrough apa adanya
 *   GET  /                                   -> halaman dokumentasi
 *   GET  /health                             -> cek upstream
 *
 * Jalankan:  bun C:/tmp/9router-get/server.mjs
 *            (atau: node --experimental-fetch C:/tmp/9router-get/server.mjs)
 * ------------------------------------------------------------------
 */

const UPSTREAM = process.env.NINE_ROUTER_URL ?? "https://9router-production-fc96.up.railway.app/v1";
const API_KEY = process.env.NINE_ROUTER_KEY ?? "";
if (!API_KEY) {
  console.error("NINE_ROUTER_KEY belum di-set. Contoh: NINE_ROUTER_KEY=sk-... bun server.mjs");
  process.exit(1);
}
const PORT = Number(process.env.PORT ?? 8787);
const DEFAULT_MODEL = process.env.NINE_ROUTER_MODEL ?? "de/92r-cbai/deepseek-v4.1-flash";

// Upstream ini suka balas 403 "temporarily blocked" / 404 route hilang
// untuk sebagian model -> coba beberapa kandidat secara berurutan.
const FALLBACKS = (process.env.NINE_ROUTER_FALLBACKS ?? "gcli/grok-4.6").split(",").map((s) => s.trim()).filter(Boolean);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

const log = (...a) => console.log(new Date().toISOString(), ...a);

/** Panggil upstream chat/completions (non-stream) dan kembalikan objek JSON. */
async function callUpstream({ model, messages, max_tokens = 1024, temperature, extra = {} }) {
  const body = {
    model,
    messages,
    max_tokens,
    ...(temperature === undefined ? {} : { temperature }),
    ...extra,
  };

  const t0 = Date.now();
  const res = await fetch(`${UPSTREAM}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(120_000),
  });

  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = { error: { message: text.slice(0, 500) } };
  }

  const elapsed = Date.now() - t0;
  if (!res.ok || json.error) {
    const msg = json?.error?.message ?? `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    err.retryable = /blocked|not found|unavailable|reset after|rate ?limit|429|5\d\d/i.test(msg);
    err.elapsed = elapsed;
    throw err;
  }

  const content = json.choices?.[0]?.message?.content ?? "";
  const reasoning = json.choices?.[0]?.message?.reasoning_content ?? "";
  return { json, content, reasoning, elapsed, modelUsed: json.model ?? model };
}

/** Coba model utama, lalu fallback kalau error-nya "retryable". */
async function ask({ model, messages, max_tokens, temperature, extra }) {
  const chain = [...new Set([model, ...FALLBACKS])];
  let last;
  for (const m of chain) {
    try {
      const r = await callUpstream({ model: m, messages, max_tokens, temperature, extra });
      return { ...r, requestedModel: model };
    } catch (e) {
      last = e;
      log(`[warn] model=${m} gagal: ${e.message}`);
    }
  }
  throw last;
}

function buildMessages({ q, system, history }) {
  const msgs = [];
  if (system) msgs.push({ role: "system", content: system });
  for (const h of history ?? []) {
    if (h && h.role && typeof h.content === "string") msgs.push({ role: h.role, content: h.content });
  }
  msgs.push({ role: "user", content: q });
  return msgs;
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS },
  });
}

const HELP = `9router GET-shim — upstream: ${UPSTREAM}

ENDPOINT GET (ini yang bikin prompt bisa di-embed lewat URL biasa)

  /ask?q=PROMPT[&model=...][&system=...][&max_tokens=...][&temperature=...]
      -> text/plain, isi = jawaban model apa adanya. Paling enak buat <iframe>/curl.

  /v1/chat/completions?q=PROMPT[&model=...][&system=...]
      -> JSON bentuk OpenAI + field "context" (messages yang dikirim) dan "answer".

  /v1/models            -> daftar model upstream
  /health               -> status upstream + model default
  /                     -> halaman ini

ENDPOINT POST (drop-in base URL, tidak diubah)

  POST /v1/chat/completions   -> diteruskan apa adanya ke upstream (stream juga jalan)

CONTOH

  curl "http://localhost:${PORT}/ask?q=jelaskan+HTTP+405"
  curl "http://localhost:${PORT}/v1/chat/completions?q=halo&model=gcli/grok-4.6"
  <iframe src="http://localhost:${PORT}/ask?q=bikin+5+rekomendasi+judul"></iframe>

Default model: ${DEFAULT_MODEL}
Fallback    : ${FALLBACKS.join(", ") || "(tidak ada)"}
`;

const server = Bun.serve({
  port: PORT,
  idleTimeout: 255,
  async fetch(req) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    // ---------- passthrough POST (drop-in) ----------
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      const upstream = await fetch(`${UPSTREAM}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
        body: req.body,
        duplex: "half",
      });
      return new Response(upstream.body, {
        status: upstream.status,
        headers: {
          ...CORS,
          "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        },
      });
    }

    if (req.method !== "GET") return json({ error: { message: "Method not allowed" } }, 405);

    // ---------- GET /health ----------
    if (url.pathname === "/health") {
      let ok = false;
      let detail = "";
      try {
        const r = await fetch(`${UPSTREAM}/models`, {
          headers: { Authorization: `Bearer ${API_KEY}` },
          signal: AbortSignal.timeout(15_000),
        });
        ok = r.ok;
        detail = `HTTP ${r.status}`;
      } catch (e) {
        detail = e.message;
      }
      return json({ ok, upstream: UPSTREAM, detail, default_model: DEFAULT_MODEL, fallbacks: FALLBACKS });
    }

    // ---------- GET /v1/models ----------
    if (url.pathname === "/v1/models") {
      const r = await fetch(`${UPSTREAM}/models`, {
        headers: { Authorization: `Bearer ${API_KEY}` },
        signal: AbortSignal.timeout(30_000),
      });
      return new Response(r.body, {
        status: r.status,
        headers: { ...CORS, "Content-Type": "application/json; charset=utf-8" },
      });
    }

    // ---------- GET /ask  &  GET /v1/chat/completions ----------
    if (url.pathname === "/ask" || url.pathname === "/v1/chat/completions") {
      const p = url.searchParams;
      const q = p.get("q") ?? p.get("prompt") ?? p.get("query") ?? p.get("message");
      if (!q) {
        return url.pathname === "/ask"
          ? new Response(HELP, { status: 400, headers: { "Content-Type": "text/plain; charset=utf-8", ...CORS } })
          : json({ error: { message: "parameter 'q' wajib diisi" } }, 400);
      }

      const model = p.get("model") ?? DEFAULT_MODEL;
      const system = p.get("system") ?? undefined;
      let history;
      if (p.get("history")) {
        try {
          history = JSON.parse(p.get("history"));
        } catch {
          return json({ error: { message: "parameter 'history' harus JSON array" } }, 400);
        }
      }

      const max_tokens = Number(p.get("max_tokens") ?? 1024);
      const temperature = p.get("temperature") === null ? undefined : Number(p.get("temperature"));

      let out;
      try {
        out = await ask({ model, messages: buildMessages({ q, system, history }), max_tokens, temperature });
      } catch (e) {
        const payload = { error: { message: e.message, status: e.status ?? 502 } };
        return url.pathname === "/ask"
          ? new Response(`[ERROR] ${e.message}\n`, {
              status: e.status && e.status >= 400 && e.status < 500 ? e.status : 502,
              headers: { "Content-Type": "text/plain; charset=utf-8", ...CORS },
            })
          : json(payload, 502);
      }

      log(`GET ${url.pathname} model=${out.modelUsed} ${out.elapsed}ms len=${out.content.length}`);

      if (url.pathname === "/ask") {
        return new Response(out.content, {
          headers: { "Content-Type": "text/plain; charset=utf-8", ...CORS },
        });
      }

      // JSON ala OpenAI + context
      return json({
        id: out.json.id ?? `chatcmpl-get-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: out.modelUsed,
        requested_model: out.requestedModel,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: out.content,
              ...(out.reasoning ? { reasoning_content: out.reasoning } : {}),
            },
            finish_reason: "stop",
          },
        ],
        answer: out.content,
        context: {
          upstream: UPSTREAM,
          model: out.modelUsed,
          elapsed_ms: out.elapsed,
          messages: buildMessages({ q, system, history }),
        },
        usage: out.json.usage ?? null,
      });
    }

    // ---------- GET /demo (halaman uji embed, same-origin) ----------
    if (url.pathname === "/demo" || url.pathname === "/demo.html") {
      const f = Bun.file(new URL("./demo.html", import.meta.url));
      return new Response(f, { headers: { "Content-Type": "text/html; charset=utf-8", ...CORS } });
    }

    // ---------- GET / ----------
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(HELP, { headers: { "Content-Type": "text/plain; charset=utf-8", ...CORS } });
    }

    return json({ error: { message: `route tidak dikenal: ${url.pathname}`, hint: `GET / untuk dokumentasi` } }, 404);
  },
});

log(`9router GET-shim listen http://localhost:${server.port}`);
log(`upstream=${UPSTREAM} default_model=${DEFAULT_MODEL}`);
