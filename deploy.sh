#!/bin/bash
# Deploy script for UVIP-AI to RunPod RTX 4090

set -e  # Exit on error

echo "🔨 Building Docker image..."
docker build -t uvip-ai:latest .

echo "📦 Tagging image for RunPod registry..."
runpod_user=$(docker login runpod.io --username YOUR_USER_ID | grep "Username:" | awk '{print $1}')  # Replace this later
docker tag uvip-ai:latest runpod.io/$runpod_user/uvip-ai:latest

echo "🚀 Pushing to RunPod registry..."
docker push runpod.io/$runpod_user/uvip-ai:latest

echo "🐳 Creating RunPod pod..."
runpodctl create pod \
  --name uvip-production \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --volumeSize 200 \
  --imageName runpod.io/$runpod_user/uvip-ai:latest

echo "✅ Deployment complete!"
echo "⏳ Waiting for pod to be ready..."
sleep 30

echo "Getting endpoint IP..."
runpodctl get endpoints --name uvip-production
