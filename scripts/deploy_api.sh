#!/bin/bash
# UVIP AI - Restart API server + BUKTIKAN versi kode yang jalan.
#
# Pakai:
#   bash scripts/deploy_api.sh              # pull + restart + verifikasi
#   SKIP_PULL=1 bash scripts/deploy_api.sh  # restart saja (tanpa git pull)
#
# Script GAGAL (exit 1) kalau build yang melayani /health bukan build di source.

set -uo pipefail

PORT="${PORT:-8001}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Repo: $ROOT"
echo "==> Port: $PORT"

# --- python yang benar-benar punya uvicorn -------------------------------
PYTHON=""
for ACT in venv/bin/activate .venv/bin/activate venv/Scripts/activate .venv/Scripts/activate; do
    if [ -f "$ACT" ]; then
        # shellcheck disable=SC1090
        source "$ACT" >/dev/null 2>&1 || true
        echo "==> venv aktivasi dicoba: $ACT"
        break
    fi
done

for C in venv/bin/python .venv/bin/python venv/Scripts/python.exe .venv/Scripts/python.exe; do
    if [ -x "$C" ] && "$C" -c "import uvicorn" >/dev/null 2>&1; then
        PYTHON="$C"; break
    fi
done
if [ -z "$PYTHON" ] && command -v python >/dev/null 2>&1 && python -c "import uvicorn" >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
fi
if [ -z "$PYTHON" ]; then
    echo "!! Tidak ada python dengan uvicorn. Aktifkan venv / pip install -r requirements.txt"; exit 1
fi
echo "==> Python: $PYTHON ($("$PYTHON" -V 2>&1))"

# --- pull ---------------------------------------------------------------
if [ "${SKIP_PULL:-0}" != "1" ]; then
    BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
    echo "==> git pull origin $BRANCH"
    git pull --ff-only origin "$BRANCH" || { echo "!! git pull gagal"; exit 1; }
fi
echo "==> HEAD: $(git log --oneline -1)"

# --- build stamp yang DIHARAPKAN (dibaca dari source, bukan hardcode) ----
EXPECTED="$(sed -n 's/^BUILD_STAMP *= *"\([^"]*\)".*/\1/p' src/uvip_ai/api/main.py | head -n 1)"
if [ -z "$EXPECTED" ]; then
    echo "!! BUILD_STAMP tidak ditemukan di src/uvip_ai/api/main.py"; exit 1
fi
echo "==> Build stamp di source: $EXPECTED"

# --- stop proses lama ---------------------------------------------------
SYSTEMD=0
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -qE '^uvip\.service'; then
    SYSTEMD=1
fi

if [ "$SYSTEMD" = "1" ]; then
    echo "==> systemd unit uvip.service terdeteksi -> pakai systemctl restart"
    systemctl restart uvip || { echo "!! systemctl restart uvip gagal"; exit 1; }
    NEW_PID="$(systemctl show -p MainPID --value uvip 2>/dev/null || true)"
    echo "==> PID baru (MainPID): ${NEW_PID:-unknown}"
else
    PIDS="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $NF}' \
            | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u || true)"
    PIDS="$PIDS $(pgrep -f "uvicorn.*uvip_ai.api.main" || true)"
    PIDS="$(echo "$PIDS" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)"
    if [ -n "$PIDS" ]; then
        echo "==> Matikan proses lama: $(echo "$PIDS" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill $PIDS 2>/dev/null || true
        sleep 2
        # shellcheck disable=SC2086
        kill -9 $PIDS 2>/dev/null || true
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "!! Port $PORT masih terpakai - hentikan manual dulu"; exit 1
    fi

    # --- start ----------------------------------------------------------
    echo "==> Start uvicorn"
    PYTHONPATH="$ROOT/src" nohup "$PYTHON" -m uvicorn uvip_ai.api.main:app \
        --host 0.0.0.0 --port "$PORT" > uvicorn.log 2>&1 &
    NEW_PID=$!
    echo "==> PID baru: $NEW_PID"
fi

# --- tunggu + verifikasi build yang benar-benar melayani -----------------
for i in $(seq 1 60); do
    LIVE="$(curl -s --max-time 3 "http://localhost:$PORT/health" \
            | sed -n 's/.*"build" *: *"\([^"]*\)".*/\1/p' || true)"
    if [ -n "$LIVE" ]; then break; fi
    sleep 1
done

if [ -z "${LIVE:-}" ]; then
    echo "!! /health tidak merespons. 40 baris terakhir log:"
    if [ "$SYSTEMD" = "1" ]; then
        journalctl -u uvip -n 40 --no-pager || true
    else
        tail -n 40 uvicorn.log
    fi
    exit 1
fi

echo "==> /health build yang JALAN: $LIVE"
if [ "$LIVE" != "$EXPECTED" ]; then
    echo "!! MISMATCH: yang jalan '$LIVE', di source '$EXPECTED'."
    echo "!! Proses lama masih melayani, atau kode di server belum ter-update."
    echo "!! Cek: ss -ltnp | grep :$PORT ; pgrep -af uvicorn"
    exit 1
fi
echo "OK: build cocok"

# --- diagnosa ffmpeg ----------------------------------------------------
echo "==> Diagnosa ffmpeg"
curl -s --max-time 120 "http://localhost:$PORT/health/ffmpeg" | "$PYTHON" -m json.tool \
    || echo "(endpoint /health/ffmpeg tidak ada - berarti masih build lama)"

if [ "$SYSTEMD" = "1" ]; then
    echo "==> Selesai. Log: journalctl -u uvip -f"
else
    echo "==> Selesai. Log: tail -f $ROOT/uvicorn.log"
fi
