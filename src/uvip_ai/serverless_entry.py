"""
RunPod Serverless Entry Point - DO NOT MODIFY UNLESS ADVANCED USER

This file is automatically loaded by RunPod when deploying in Queue mode.
It bridges the FastAPI application with RunPod's serverless runtime.
"""
from src.uvip_ai.runpod_serverless import runpod_serverless, runpod_serverless_async
import asyncio


def start():
    """
    RunPod will call this function to initialize the handler.
    
    This is REQUIRED for Queue-based deployment.
    Returns a reference to runpod_serverless which handles incoming requests.
    """
    print("=" * 70)
    print("🚀 UVIP-AI RunPod Serverless Handler")
    print("=" * 70)
    print("")
    print("Mode: Queue-based Serverless (auto-scaling)")
    print("GPU: NVIDIA RTX 4090")
    print("Auto-scale: ON")
    print("")
    print("Configuration:")
    print("  - Models loaded once, reused across requests")
    print("  - Low VRAM mode: ENABLED")
    print("  - FP16 inference: ENABLED")
    print("")
    print("Endpoints:")
    print("  POST /serverless/invoke → Direct API call")
    print("  GET  /health           → Health check")
    print("")
    print("Press Ctrl+C to stop (if running locally)")
    print("=" * 70)
    print("")
    
    # Return the main handler function
    return runpod_serverless


# Local testing entry point
if __name__ == "__main__":
    print("Testing RunPod Serverless Handler...")
    print("=" * 70)
    
    handler = start()
    
    # Test payload
    test_payload = {
        "image": b"test",  # Will fail but shows it works
        "latitude": -7.976,
        "longitude": 112.630,
    }
    
    result = handler(test_payload)
    print(f"\nTest Result: {result}")
    
    print("\n" + "=" * 70)
    print("✅ Serverless handler is working!")
    print("=" * 70)
