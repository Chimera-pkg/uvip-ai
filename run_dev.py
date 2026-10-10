#!/usr/bin/env python3
"""Jalankan API UVIP-AI untuk development lokal.

Pakai:
    .venv\\Scripts\\python.exe run_dev.py        (Windows)
    .venv/bin/python run_dev.py                  (Linux/macOS)

Opsi:
    --host 0.0.0.0    bind ke semua interface (default 127.0.0.1)
    --port 8010       ganti port (default 8001)
    --reload          auto-restart saat file berubah (dev)

Tidak perlu set PYTHONPATH: script ini menambahkan `src/` sendiri, jadi
`import uvip_ai` selalu ketemu walau paket belum di-install.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# `import uvip_ai` harus jalan baik saat dijalankan langsung maupun di
# subprocess reload uvicorn (yang mewarisi environment).
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
_pp = os.environ.get("PYTHONPATH", "")
if str(SRC) not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = str(SRC) + (os.pathsep + _pp if _pp else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan API UVIP-AI (dev)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true",
                        help="auto-restart saat file berubah (dev)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("!! uvicorn belum terpasang. Jalankan: pip install -r requirements.txt")
        sys.exit(1)

    # Pastikan model & import bisa dimuat sebelum bind port.
    import uvip_ai  # noqa: F401  (fail cepat kalau dependency kurang)

    print(f"==> UVIP-AI API: http://{args.host}:{args.port}")
    print(f"==> Docs      : http://{args.host}:{args.port}/docs")
    print(f"==> Health    : http://{args.host}:{args.port}/health")
    print("==> Ctrl+C untuk berhenti")

    uvicorn.run("uvip_ai.api.main:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    main()
