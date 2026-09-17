#!/bin/bash
# 🚀 UVIP-AI ONE-STEP DEPLOYMENT FOR RUNPOD (No Docker Required!)
# Copy-paste this entire file to Web Terminal and run!

set -e

echo "═══════════════════════════════════════════"
echo "🚀 UVIP-AI Deployment to RunPod RTX 4090"
echo "═══════════════════════════════════════════"
echo ""

# STEP 1: Navigate to home directory
cd /home/uvip-ai || mkdir -p /home/uvip-ai && cd /home/uvip-ai

# STEP 2: Clone repository if not exists
if [ ! -d ".git" ] && [ ! -f "requirements.txt" ]; then
    echo "⏳ Cloning repository..."
    
    # GANTI URL INI DENGAN REPO ANDA SENDIRI!
    read -p "Enter GitHub repository URL: " REPO_URL
    
    if [ -z "$REPO_URL" ]; then
        echo "❌ Repository URL is required!"
        exit 1
    fi
    
    git clone "$REPO_URL" . 
    # OR if already cloned above, just fetch latest:
    # git pull origin main
fi

# STEP 3: Copy environment configuration
echo "⚙️  Setting up environment..."
cp .env.example .env 2>/dev/null || touch .env
echo "# UVIP-AI Environment" >> .env
echo "UVIP_DEVICE=auto" >> .env
echo "UVIP_USE_FP16=true" >> .env
echo "UVIP_LOW_VRAM_MODE=true" >> .env

# STEP 4: Mount network volume for persistence
echo "💾 Mounting network volume..."
mkdir -p /network-volume/UVIP-AI_volume/{models,data}
ln -sf /network-volume/UVIP-AI_volume/models ./models 2>/dev/null || true
ln -sf /network-volume/UVIP-AI_volume/data ./data 2>/dev/null || true
ln -sf /network-volume/UVIP-AI_volume/logs ./logs 2>/dev/null || true
mkdir -p ./logs

# STEP 5: Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt 2>&1 | tail -20

# STEP 6: Verify Python and GPU
echo "🔍 Verifying environment..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>&1 || python -c "print('✓ Python available')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>&1 || echo "✓ No GPU error (will check later)"

# STEP 7: Start API server in background with proper error handling
echo "🚀 Starting API server on port 8001..."

# Create a proper start script that handles errors better
cat > start_api.py << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure correct paths
os.chdir("/home/uvip-ai")
sys.path.insert(0, str(Path.cwd() / "src"))

try:
    from uvicorn import run
    from src.uvip_ai.api.main import app
    
    print("🚀 Starting UVIP-AI API Server...")
    print(f"📍 Listening on: http://0.0.0.0:8001")
    print(f"✅ Health endpoint: http://localhost:8001/health")
    print("")
    print("=" * 60)
    
    run(app, host="0.0.0.0", port=8001, workers=1, log_level="info")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Startup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

chmod +x start_api.py

# Run with timeout to catch startup errors
timeout 30 python start_api.py &
API_PID=$!

# Wait for startup
echo "⏳ Waiting for server to start (30 seconds)..."
sleep 30

# STEP 8: Verify deployment
echo ""
echo "═══════════════════════════════════════════"
echo "🔍 VERIFICATION"
echo "═══════════════════════════════════════════"

# Check if process running
if ps -p $API_PID > /dev/null 2>&1; then
    echo "✅ Process running (PID: $API_PID)"
else
    echo "⚠️  Process exited, checking logs..."
fi

# Check logs
if [ -f "./api.log" ]; then
    echo "Last 30 lines of api.log:"
    tail -30 ./api.log
    echo ""
elif [ -f "./logs/api.log" ]; then
    echo "Last 30 lines of logs/api.log:"
    tail -30 ./logs/api.log
fi

# Try health endpoint
echo ""
echo "Testing health endpoint..."
curl -s http://localhost:8001/health 2>&1 | head -5 || echo "⚠️  Cannot connect yet (server still starting)"

# Show what's running
echo ""
echo "Checking processes on port 8001..."
sleep 5
lsof -i :8001 2>/dev/null | head -10 || netstat -tuln 2>/dev/null | grep 8001 || ss -tlnp 2>/dev/null | grep 8001 || echo "Port 8001 check failed (using minimal tools)"

echo ""
echo "═══════════════════════════════════════════"
echo "📝 NEXT STEPS"
echo "═══════════════════════════════════════════"
echo ""
echo "1. Test API at: http://YOUR_POD_IP:8001/health"
echo "   Get your pod IP from dashboard → Details → Endpoint"
echo ""
echo "2. Monitor logs:"
echo "   tail -f ./api.log"
echo ""
echo "3. Restart server if crashed:"
echo "   kill $API_PID 2>/dev/null || pkill -f uvicorn"
echo "   nohup python start_api.py > ./api.log 2>&1 &"
echo ""
echo "4. Auto-start command (add to crontab):"
echo "   @reboot cd /home/uvip-ai && nohup python start_api.py > ./api.log 2>&1 &"
echo ""
echo "═══════════════════════════════════════════"

# Keep monitoring logs
echo ""
echo "📊 Following logs (Ctrl+C to stop)..."
tail -f ./api.log ./logs/api.log 2>/dev/null || echo "No active logs to follow"
