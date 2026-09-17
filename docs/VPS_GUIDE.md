# 💰 Panduan VPS Murah untuk UVIP AI

## 🎯 Rekomendasi VPS Paling Murah

### ✅ **Pilihan #1: Contabo (PALING MURAH)**
**Harga: $6.99/bulan (~Rp 110.000)**

| Spesifikasi | Detail |
|-------------|--------|
| **CPU** | 4 vCPU |
| **RAM** | 8 GB |
| **Storage** | 200 GB NVMe |
| **Bandwidth** | Unlimited |
| **Location** | Germany/USA/Singapore/Japan |

**Link:** https://contabo.com/en/cloud-vps/

**Kelebihan:**
- ✅ Paling murah untuk spek gede
- ✅ RAM 8GB cukup untuk UVIP
- ✅ Storage 200GB lega
- ✅ Unlimited bandwidth

**Kekurangan:**
- ⚠️ Support lambat
- ⚠️ Network speed biasa

---

### ✅ **Pilihan #2: Hetzner (BEST VALUE)**
**Harga: €4.85/bulan (~Rp 85.000)**

| Spesifikasi | Detail |
|-------------|--------|
| **CPU** | 2 vCPU |
| **RAM** | 4 GB |
| **Storage** | 40 GB SSD |
| **Bandwidth** | 20 TB |
| **Location** | Germany/Finland/USA |

**Link:** https://www.hetzner.com/cloud

**Kelebihan:**
- ✅ Sangat murah
- ✅ Performance bagus
- ✅ Network cepat
- ✅ Reliable

**Kekurangan:**
- ⚠️ RAM 4GB (minimal untuk UVIP)
- ⚠️ Server di Eropa (latency ke Indonesia)

---

### ✅ **Pilihan #3: DigitalOcean (RELIABLE)**
**Harga: $12/bulan (~Rp 190.000)**

| Spesifikasi | Detail |
|-------------|--------|
| **CPU** | 2 vCPU |
| **RAM** | 4 GB |
| **Storage** | 80 GB SSD |
| **Bandwidth** | 4 TB |
| **Location** | Singapore (dekat Indonesia!) |

**Link:** https://www.digitalocean.com/pricing

**Kelebihan:**
- ✅ Server Singapore (latency rendah)
- ✅ Documentation bagus
- ✅ Reliable
- ✅ Easy to use

**Kekurangan:**
- ⚠️ Lebih mahal dari Contabo

---

### ✅ **Pilihan #4: Vultr (JAKARTA!)**
**Harga: $12/bulan (~Rp 190.000)**

| Spesifikasi | Detail |
|-------------|--------|
| **CPU** | 1 vCPU |
| **RAM** | 2 GB |
| **Storage** | 50 GB SSD |
| **Bandwidth** | 2 TB |
| **Location** | **Jakarta, Indonesia** |

**Link:** https://www.vultr.com/pricing/

**Kelebihan:**
- ✅ **Server di Jakarta** (paling cepat!)
- ✅ Latency sangat rendah
- ✅ Hourly billing

**Kekurangan:**
- ⚠️ RAM 2GB (kurang untuk UVIP)
- ⚠️ Harus upgrade ke 4GB ($24/bulan)

---

## 📊 Perbandingan Lengkap

| Provider | Harga | CPU | RAM | Storage | Location | Best For |
|----------|-------|-----|-----|---------|----------|----------|
| **Contabo** | **$7** | 4 vCPU | 8GB | 200GB | Global | **Budget** ✅ |
| **Hetzner** | **€5** | 2 vCPU | 4GB | 40GB | EU/US | **Value** ✅ |
| **DigitalOcean** | $12 | 2 vCPU | 4GB | 80GB | Singapore | **Reliable** |
| **Vultr** | $24 | 2 vCPU | 4GB | 80GB | **Jakarta** | **Speed** ✅ |

---

## 🎯 Rekomendasi Saya

### **Untuk Budget Ketat: Contabo $7/bulan**
```
4 vCPU, 8GB RAM, 200GB Storage
= Rp 110.000/bulan
```
**Cukup untuk:**
- ✅ Backend API (FastAPI)
- ✅ Database (PostgreSQL)
- ✅ File storage
- ✅ Inference ringan (CPU mode)

**Tidak cukup untuk:**
- ❌ Training model (butuh GPU)
- ❌ Inference berat (butuh GPU)

---

### **Untuk Performance: Hetzner €5/bulan**
```
2 vCPU, 4GB RAM, 40GB Storage
= Rp 85.000/bulan
```
**Cukup untuk:**
- ✅ Backend API
- ✅ Database
- ✅ Inference ringan

---

### **Untuk Speed (Indonesia): Vultr $24/bulan**
```
2 vCPU, 4GB RAM, 80GB Storage
= Rp 380.000/bulan
```
**Kelebihan:**
- ✅ Server Jakarta (latency <10ms)
- ✅ Cepat untuk user Indonesia

---

## 🧠 Berapa Core Minimal untuk UVIP?

### **Minimum Requirements:**

| Komponen | CPU | RAM | Storage |
|----------|-----|-----|---------|
| **Backend API** | 1 vCPU | 1 GB | 10 GB |
| **Database** | 1 vCPU | 1 GB | 20 GB |
| **Inference (CPU)** | 2 vCPU | 4 GB | 50 GB |
| **TOTAL MINIMAL** | **2 vCPU** | **4 GB** | **50 GB** |

### **Recommended:**

| Komponen | CPU | RAM | Storage |
|----------|-----|-----|---------|
| **Backend API** | 2 vCPU | 2 GB | 20 GB |
| **Database** | 2 vCPU | 2 GB | 50 GB |
| **Inference (CPU)** | 4 vCPU | 8 GB | 100 GB |
| **TOTAL RECOMMENDED** | **4 vCPU** | **8 GB** | **100 GB** |

---

## 🚀 Setup UVIP di VPS

### **Step 1: Beli VPS**

**Rekomendasi: Contabo $7/bulan**
1. Buka https://contabo.com/en/cloud-vps/
2. Pilih "Cloud VPS S" ($6.99/bulan)
3. Pilih location: **Singapore** (dekat Indonesia)
4. Pilih OS: **Ubuntu 22.04 LTS**
5. Checkout & bayar

### **Step 2: Connect ke VPS**

```bash
# SSH ke VPS
ssh root@YOUR_VPS_IP

# Update system
apt update && apt upgrade -y

# Install dependencies
apt install -y python3-pip python3-venv git curl nginx
```

### **Step 3: Clone Repository**

```bash
# Clone UVIP AI
cd /opt
git clone https://github.com/your-username/uvip-ai.git
cd uvip-ai

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 4: Setup Backend**

```bash
# Copy trained models dari laptop
# (upload dari laptop ke VPS)
scp -r models/ root@YOUR_VPS_IP:/opt/uvip-ai/

# Test backend
python scripts/api_server.py
```

### **Step 5: Setup Nginx**

```bash
# Create nginx config
cat > /etc/nginx/sites-available/uvip << 'EOF'
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /uploads/ {
        alias /opt/uvip-ai/uploads/;
        autoindex on;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/uvip /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

### **Step 6: Setup Systemd Service**

```bash
# Create service file
cat > /etc/systemd/system/uvip.service << 'EOF'
[Unit]
Description=UVIP AI Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/uvip-ai
Environment="PATH=/opt/uvip-ai/venv/bin"
ExecStart=/opt/uvip-ai/venv/bin/python scripts/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable & start service
systemctl enable uvip
systemctl start uvip
systemctl status uvip
```

### **Step 7: Setup Firewall**

```bash
# Install firewall
apt install ufw -y

# Allow SSH, HTTP, HTTPS
ufw allow 22
ufw allow 80
ufw allow 443

# Enable firewall
ufw enable
```

### **Step 8: Test**

```bash
# Test API
curl http://YOUR_VPS_IP/health

# Test prediction
curl -X POST http://YOUR_VPS_IP/predict \
  -F "file=@test_image.jpg"
```

---

## 💸 Total Biaya

### **Opsi 1: Contabo (Paling Murah)**
```
VPS: $7/bulan (Rp 110.000)
Domain: $10/tahun (Rp 160.000) - optional
SSL: Free (Let's Encrypt)
Total: Rp 110.000/bulan
```

### **Opsi 2: Hetzner (Best Value)**
```
VPS: €5/bulan (Rp 85.000)
Domain: $10/tahun (Rp 160.000) - optional
SSL: Free (Let's Encrypt)
Total: Rp 85.000/bulan
```

### **Opsi 3: Vultr Jakarta (Fastest)**
```
VPS: $24/bulan (Rp 380.000)
Domain: $10/tahun (Rp 160.000) - optional
SSL: Free (Let's Encrypt)
Total: Rp 380.000/bulan
```

---

## 🎯 Kesimpulan

### **VPS Paling Murah: Contabo $7/bulan**
- ✅ 4 vCPU, 8GB RAM, 200GB Storage
- ✅ Cukup untuk UVIP backend + inference ringan
- ✅ Rp 110.000/bulan

### **VPS Terbaik: Hetzner €5/bulan**
- ✅ 2 vCPU, 4GB RAM, 40GB Storage
- ✅ Performance bagus, reliable
- ✅ Rp 85.000/bulan

### **VPS Tercepat: Vultr Jakarta $24/bulan**
- ✅ Server di Jakarta
- ✅ Latency <10ms ke Indonesia
- ✅ Rp 380.000/bulan

---

## 📝 Next Steps

1. **Pilih VPS** (rekomendasi: Contabo $7/bulan)
2. **Beli VPS** (5 menit)
3. **Setup UVIP** (30 menit)
4. **Deploy backend** (15 menit)
5. **Test API** (5 menit)

**Total waktu:** ~1 jam
**Total biaya:** Rp 110.000/bulan

---

Mau saya buatkan script deployment otomatis untuk VPS? Tinggal copy-paste ke terminal dan VPS langsung jalan!
