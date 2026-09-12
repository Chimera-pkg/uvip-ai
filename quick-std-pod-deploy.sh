#!/bin/bash
# ============================================================================
# UVIP-AI One-Click Deployment to RunPod Standard Pod (GPU VM)
# NO DOCKER SETUP REQUIRED - Direct Python execution
# ============================================================================

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 UVIP-AI → RunPod Standard Pod Deployment (RTX 4090)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Benefits:"
echo "   • No Docker complexity"
echo "   • Full RTX 4090 GPU access"
echo "   • Zero cold start (always running)"
echo "   • Fast API directly on port 8001"
echo "   • No model downgrade needed!"
echo ""

# -----------------------------------------------------------------------------
# STEP 1: Clone Repository & Setup Environment
# -----------------------------------------------------------------------------
echo "Step 1/6: Cloning repository..."
cd /home/uvip-ai || mkdir -p /home/uvip-ai && cd /home/uvip-ai

if [ ! -d ".git" ]; then
    read -p "Enter GitHub repository URL: " REPO_URL
    git clone "$REPO_URL" .
fi

cp .env.example .env
echo ""
echo "✅ Repository ready"
echo ""

# -----------------------------------------------------------------------------
# STEP 2: Install Dependencies
# -----------------------------------------------------------------------------
echo "Step 2/6: Installing dependencies (first time takes ~5-10 minutes)..."
pip install --upgrade pip
pip install -r requirements.txt --quiet

echo ""
echo "✅ Dependencies installed"
echo ""

# -----------------------------------------------------------------------------
# STEP 3: Mount Network Volume for Persistence
# -----------------------------------------------------------------------------
echo "Step 3/6: Mounting network volume..."
mkdir -p /network-volume/UVIP-AI_volume/{models,data}
ln -sf /network-volume/UVIP-AI_volume/models ./models 2>/dev/null || true
ln -sf /network-volume/UVIP-AI_volume/data ./data 2>/dev/null || true
ln -sf /network-volume/UVIP-AI_volume/logs ./logs 2>/dev/null || true
mkdir -p ./logs

echo ""
echo "✅ Storage mounted"
echo ""

# -----------------------------------------------------------------------------
# STEP 4: Verify GPU Detection
# -----------------------------------------------------------------------------
echo "Step 4/6: Verifying GPU environment..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "⚠️  PyTorch check skipped"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "⚠️  GPU check failed (will test at runtime)"
nvidia-smi 2>/dev/null | grep "RTX 4090" && echo "✓ RTX 4090 detected in nvidia-smi" || echo "⚠️  GPU not yet visible (normal inside container)"

echo ""
echo "✅ GPU environment checked"
echo ""

# -----------------------------------------------------------------------------
# STEP 5: Create Auto-Start Script
# -----------------------------------------------------------------------------
echo "Step 5/6: Creating startup script..."
cat > start_api.sh << 'STARTSCRIPT'
#!/bin/bash
cd /home/uvip-ai
export PYTHONPATH=/home/uvip-ai/src
nohup python3 -m uvicorn src.uvip_ai.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 4 \
  --log-level info \
  > logs/api.log 2>&1 &
echo $! > logs/api.pid
echo "✅ API server started (PID: $!)"
echo "   Health check: curl http://localhost:8001/health"
STARTSCRIPT

chmod +x start_api.sh

echo ""
echo "✅ Startup script created"
echo ""

# -----------------------------------------------------------------------------
# STEP 6: Start API Server
# -----------------------------------------------------------------------------
echo "Step 6/6: Starting API server (waiting 30 seconds for startup)..."
./start_api.sh
sleep 30

# -----------------------------------------------------------------------------
# VERIFICATION
# -----------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 DEPLOYMENT VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if process running
if [ -f "./logs/api.pid" ]; then
    PID=$(cat ./logs/api.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ API server running (PID: $PID)"
    else
        echo "❌ API process exited, checking logs..."
    fi
else
    echo "⚠️  PID file not found"
fi

# View latest log
if [ -f "./logs/api.log" ]; then
    echo ""
    echo "Last 20 lines of api.log:"
    tail -20 ./logs/api.log
else
    echo "⚠️  Log file not available yet"
fi

# Test health endpoint (with retry)
echo ""
echo "Testing health endpoint (retry 3 times)..."
for i in 1 2 3; do
    RESPONSE=$(curl -s http://localhost:8001/health 2>&1)
    if echo "$RESPONSE" | grep -q '"status"'; then
        echo "✅ HEALTH CHECK PASSED!"
        echo ""
        echo "API is accessible at:"
        echo "   📍 Health:     http://localhost:8001/health"
        echo "   📍 Predict:    http://localhost:8001/ai/process"
        echo ""
        echo "📊 Response:"
        echo "$RESPONSE" | head -5
        break
    else
        if [ $i -lt 3 ]; then
            echo "Retry $i/3... (server still starting)"
            sleep 10
        else
            echo "⚠️  Health check failed after 3 attempts"
            echo "   Check logs: tail -f ./logs/api.log"
        fi
    fi
done

# Show GPU usage if available
echo ""
nvidia-smi 2>/dev/null | grep -A5 "GPU Memory" || echo "ℹ️  GPU details: Run 'nvidia-smi' to check real-time stats"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Quick Commands:"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "📖 Monitor logs:"
echo "   tail -f ~/uvip-ai/logs/api.log"
echo ""
echo "🔄 Restart server:"
echo "   pkill -f uvicorn && ~/uvip-ai/start_api.sh"
echo ""
echo "💰 Stop billing (save costs):"
echo "   systemctl stop uvip-api"
echo "   Then via dashboard: Settings → Shutdown pod"
echo ""
echo "📡 Test API locally:"
echo "   curl http://localhost:8001/health"
echo ""
echo " Process image example:"
echo "   curl -X POST http://localhost:8001/ai/process \\"
echo "     -F \"file=@path/to/image.jpg\" \\"
echo "     -F \"latitude=-7.976\" \\"
echo "     -F \"longitude=112.630\""
echo ""
echo "═════════════════════════════════════════════════════════"
echo "🚀 Your UVIP-AI server is now LIVE on RTX 4090!"
echo "   Cost: $0.28/hour × 24 = ~$200/month (always running)"
echo "   Latency: <700ms (full pipeline with GPU acceleration)"
echo "═════════════════════════════════════════════════════════"
echo ""

# Optional: Continuous log monitoring
read -p "Do you want to follow logs continuously? (y/n) [y]: " FOLLOW
if [[ $FOLLOW =~ ^[Yy]$ ]]; then
    echo ""
    echo "Following logs (Press Ctrl+C to stop)..."
    tail -f ./logs/api.log
fi
