#!/bin/bash
# UVIP-AI Deployment Script for RunPod RTX 4090
# Copy this to your pod's web terminal and run it

set -e

echo "═══════════════════════════════════════════"
echo "🚀 UVIP-AI Auto-Deployment Script"
echo "═══════════════════════════════════════════"
echo ""

# 1. Clone repository
echo "Step 1: Cloning repository..."
read -p "Enter your GitHub repo URL: " REPO_URL
if [ -z "$REPO_URL" ]; then
    echo "❌ Repository URL required!"
    exit 1
fi

git clone "$REPO_URL" uvip-ai
cd uvip-ai

# 2. Setup environment
echo ""
echo "Step 2: Setting up environment..."
cp .env.example .env

# 3. Install dependencies (optional if using Docker)
echo ""
echo "Step 3: Installing dependencies..."
pip install -r requirements.txt

# 4. Build Docker image
echo ""
echo "Step 4: Building Docker image..."
docker build -t uvip-ai:latest .

# 5. Run container
echo ""
echo "Step 5: Starting API server..."
docker run --gpus all \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v /network-volume/UVIP-AI_volume:/app/network-volume \
  --name uvip-api \
  --restart unless-stopped \
  uvip-ai:latest &

sleep 5

# 6. Verify
echo ""
echo "═══════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════"
echo ""
echo "📊 Container Status:"
docker ps | grep uvip-api

echo ""
echo "🏥 Health Check:"
curl -s http://localhost:8001/health | python -m json.tool || echo "⚠️  API still starting..."

echo ""
echo "📖 View Logs:"
echo "   docker logs -f uvip-api"

echo ""
echo "🛑 Stop Container:"
echo "   docker stop uvip-api"

echo ""
echo "💰 Cost Management:"
echo "   # Stop pod in dashboard to save costs"
echo "   # Pod status: Running (billing active)"

echo ""
echo "🌐 Access API at:"
echo "   http://YOUR_POD_IP:8001/health"
echo ""
