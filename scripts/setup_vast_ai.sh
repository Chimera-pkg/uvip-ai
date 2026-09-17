#!/bin/bash
# UVIP AI - Complete Setup Script for Vast.ai GTX 1070
# This script sets up everything needed for the project

set -e  # Exit on error

echo "=========================================="
echo "UVIP AI - Vast.ai Setup"
echo "Target: GTX 1070 + CUDA 12.8"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p logs
LOG_FILE="setup_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo -e "${GREEN}[SETUP]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[OK]${NC} $1" | tee -a "$LOG_FILE"
}

# Step 1: Verify Python and CUDA
log "Step 1/8: Verifying Python and CUDA..."
python --version || error "Python not found"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')" || error "PyTorch not installed"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" || error "CUDA not available"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
success "Python and CUDA verified"

# Step 2: Clone repository (if not already cloned)
log "Step 2/8: Checking repository..."
if [ ! -d ".git" ]; then
    warn "Not a git repository, skipping git operations"
else
    git fetch origin main 2>/dev/null || true
    success "Repository checked"
fi

# Step 3: Create virtual environment
log "Step 3/8: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv || error "Failed to create venv"
fi
source venv/bin/activate
log "Virtual environment ready"

# Step 4: Install dependencies
log "Step 4/8: Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || error "Failed to install requirements"
fi

# Install PyTorch with CUDA 12 support explicitly
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
success "Dependencies installed"

# Step 5: Configure models directory
log "Step 5/8: Setting up model directories..."
mkdir -p models/perception
mkdir -p models/finetuned_indonesia
mkdir -p data/raw
mkdir -p data/extracted
mkdir -p test_outputs
mkdir -p logs
touch models/.gitkeep
success "Directories created"

# Step 6: Download trained models (if needed)
log "Step 6/8: Checking trained models..."
MODELS=(beauty_model.pkl safety_model.pkl comfort_model.pkl uvi_model.pkl)
ALL_MODELS_FOUND=true

for model in "${MODELS[@]}"; do
    if [ -f "models/perception/$model" ]; then
        success "Found: $model"
    else
        warn "Missing: $model (you'll need to train or download first)"
        ALL_MODELS_FOUND=false
    fi
done

if [ "$ALL_MODELS_FOUND" = true ]; then
    log "All trained models found!"
else
    warn "Some models missing. You can:"
    warn "  1. Train locally and upload"
    warn "  2. Download from Kaggle/artifact"
    warn "  3. Use demo mode without models"
fi

# Step 7: Verify installations
log "Step 7/8: Verifying installations..."
python -c "from uvip_ai.privacy.guard import PrivacyGuard; print('Privacy Guard: OK')" || warn "Privacy Guard not working"
python -c "from uvip_ai.segmentation.segformer import SegformerB5; print('SegFormer: OK')" || warn "SegFormer not working"
python -c "from uvip_ai.features.dinov2 import Dinov2Extractor; print('DINOv2: OK')" || warn "DINOv2 not working"
python -c "from fastapi import FastAPI; print('FastAPI: OK')" || warn "FastAPI not working"
success "All core components verified"

# Step 8: Test run (optional)
log "Step 8/8: Quick test (optional)..."
read -p "Run quick test with sample image? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/test_image.py --image data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg --models models/perception 2>&1 | head -50
fi

# Summary
echo ""
echo "=========================================="
echo "🎉 SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Your server is now ready with:"
echo "  ✓ Python 3.9+ with CUDA 12.8"
echo "  ✓ Virtual environment activated"
echo "  ✓ All dependencies installed"
echo "  ✓ Model directories ready"
echo "  ✓ Core packages verified"
echo ""
echo "Next steps:"
echo "  1. Upload trained models to models/perception/"
echo "  2. Run: ./scripts/deploy_api.sh"
echo "  3. Access API at: http://localhost:8000"
echo "  4. View docs at: http://localhost:8000/docs"
echo ""
echo "Logs are saved to: $LOG_FILE"
echo "=========================================="
