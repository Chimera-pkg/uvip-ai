#!/bin/bash
# ============================================================
# UVIP AI - Auto Deploy Script untuk VPS Ubuntu 20.04/22.04
# Usage: bash deploy.sh
# Tested on: Ubuntu 20.04/22.04, min 2 vCPU, 4GB RAM
# ============================================================
set -e

REPO_URL="https://github.com/Chimera-pkg/uvip-ai.git"
APP_DIR="/opt/uvip-ai"
PYTHON_VER="python3"
PORT=8000

echo "======================================"
echo " UVIP AI - Auto Deploy"
echo "======================================"

# 1. Update system
echo "[1/8] Update system..."
apt-get update -qq && apt-get upgrade -y -qq

# 2. Install dependencies
echo "[2/8] Install dependencies..."
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git curl nginx \
    libgl1 libglib2.0-0 \
    ufw

# 3. Clone / update repo
echo "[3/8] Clone repository..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# 4. Python virtual environment
echo "[4/8] Setup Python environment..."
cd "$APP_DIR"
$PYTHON_VER -m venv venv
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

# 5. Setup direktori & env
echo "[5/8] Setup directories..."
mkdir -p uploads/photos uploads/masks models/perception data/training
cp -n .env.example .env 2>/dev/null || true

# Set UVIP_DEVICE=cpu karena VPS tidak punya GPU
sed -i 's/UVIP_DEVICE=auto/UVIP_DEVICE=cpu/' .env
sed -i 's/UVIP_USE_FP16=true/UVIP_USE_FP16=false/' .env

# 6. Systemd service
echo "[6/8] Setup systemd service..."
cat > /etc/systemd/system/uvip.service << EOF
[Unit]
Description=UVIP AI Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$APP_DIR/src"
ExecStart=$APP_DIR/venv/bin/uvicorn uvip_ai.api.main:app --host 0.0.0.0 --port $PORT --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable uvip
systemctl restart uvip

# 7. Nginx reverse proxy
echo "[7/8] Setup Nginx..."
cat > /etc/nginx/sites-available/uvip << EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }

    location /uploads/ {
        alias $APP_DIR/uploads/;
    }
}
EOF

ln -sf /etc/nginx/sites-available/uvip /etc/nginx/sites-enabled/uvip
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# 8. Firewall
echo "[8/8] Setup firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo ""
echo "======================================"
echo " Deploy selesai!"
echo "======================================"
echo " API running di: http://$(curl -s ifconfig.me)"
echo " Health check  : curl http://$(curl -s ifconfig.me)/health"
echo ""
echo " Upload trained models:"
echo " scp models/perception/*.pkl root@VPS_IP:$APP_DIR/models/perception/"
echo ""
echo " Cek status service:"
echo " systemctl status uvip"
echo " journalctl -u uvip -f"
echo "======================================"
