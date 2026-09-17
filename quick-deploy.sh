#!/bin/bash
# Quick Deployment Script for UVIP-AI to RunPod RTX 4090
# This creates a Standard Pod (NOT Serverless/Queue mode)

set -e

echo "🔧 UVIP-AI Quick Deployment to RunPod RTX 4090"
echo "================================================"
echo ""

# Check if API key is configured
if [ ! -f ~/.runpodrc ]; then
    echo "⚠️  No API key found. Please run:"
    echo "   pip install runpodctl"
    echo "   runpodctl config --apiKey YOUR_API_KEY_HERE"
    echo ""
    read -p "Press Enter after running the above..." || exit 1
fi

# Get pod name
read -p "Enter pod name (default: uvip-production): " POD_NAME
POD_NAME=${POD_NAME:-uvip-production}

echo ""
echo " Creating Standard Pod..."
echo "   Name: $POD_NAME"
echo "   GPU: NVIDIA GeForce RTX 4090"
echo "   Storage: 200 GB"
echo ""

# Create pod with base image (NOT serverless!)
runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --volumeSize 200 \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda12.1"

echo ""
echo "✅ Pod created successfully!"
echo ""
echo "⏳ Waiting for pod to be 'Running' state..."
echo ""

# Wait for pod to be ready
sleep 30

echo "Getting pod endpoint information..."
runpodctl get endpoints --name "$POD_NAME"

echo ""
echo "═══════════════════════════════════════════"
echo "🚀 DEPLOYMENT READY!"
echo "═══════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Go to: https://www.runpod.io/console/serverless"
echo "2. Click on pod '$POD_NAME'"
echo "3. Click 'Terminal' button"
echo "4. Run these commands in terminal:"
echo ""
echo "   # Clone your repository"
echo "   git clone <your-repo-url>"
echo "   cd uvip-ai"
echo ""
echo "   # Build and run"
echo "   cp .env.example .env"
echo "   docker-compose up -d --build"
echo ""
echo "5. Test health endpoint:"
echo "   curl http://localhost:8001/health"
echo ""
echo "💰 COST SAVING TIPS:"
echo "- Stop pod when not in use: runpodctl stop --name $POD_NAME"
echo "- Start again: runpodctl start --name $POD_NAME"
echo "- Delete permanently: runpodctl delete --name $POD_NAME"
echo ""
