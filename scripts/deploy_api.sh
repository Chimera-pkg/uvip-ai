#!/bin/bash
# UVIP AI - Auto-deploy API Server to Vast.ai GTX 1070
# Automatically starts the FastAPI server with GPU support

set -e

echo "=========================================="
echo "UVIP AI - Deploying to Vast.ai GTX 1070"
echo "GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "CUDA: $(python -c "import torch; print(torch.version.cuda)")"
echo "=========================================="

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/app:$PYTHONPATH

# Create necessary directories
mkdir -p logs test_outputs models/perception

# Start Gunicorn with GPU workers
echo "Starting API server on port 8000..."
echo "Use CTRL+C to stop (server continues in background)"

nohup gunicorn src.api.main:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --log-level info \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    > api_server.log 2>&1 &

PID=$!
echo "✅ Server started with PID: $PID"
echo ""
echo "Server Status:"
echo "  📡 API URL: http://localhost:8000"
echo "  📄 API Docs: http://localhost:8000/docs"
echo "  🔍 Health Check: http://localhost:8000/health"
echo ""
echo "Logs:"
echo "  tail -f logs/access.log  # Access logs"
echo "  tail -f logs/error.log   # Error logs"
echo "  cat api_server.log       # Combined output"
echo ""
echo "Press CTRL+C to stop server"
echo "=========================================="

# Keep process running
trap "kill $PID; echo 'Server stopped'" INT TERM EXIT

while true; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "❌ Server crashed!"
        cat api_server.log
        exit 1
    fi
    sleep 30
done
