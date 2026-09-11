# -*- mode: python; coding: utf-8 -*-
# Add this code BEFORE line 539 (before if __name__) in src/uvip_ai/api/main.py

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  ADD THESE ROUTES FOR RUNPOD QUEUE-BASED SERVERLESS                        │
└─────────────────────────────────────────────────────────────────────────────┘
"""

from fastapi.responses import JSONResponse
from fastapi import HTTPException

# Root handler for RunPod Queue-based Serverless endpoints
@app.post("/")
async def serverless_root_endpoint(payload: dict = None):
    """
    ROOT-level endpoint that handles all POST requests from RunPod Queue mode.
    
    This is REQUIRED because RunPod Queue endpoints send requests to 
    https://api.runpod.ai/v2/{endpoint_id}/runsync with payload as body.
    
    The payload format expected:
    {
        "image": "",  # hex-encoded image bytes or base64
        "latitude": -7.976,
        "longitude": 112.630,
        "filename": "optional.jpg"
    }
    """
    
    if payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No payload provided",
                "success": False
            }
        )
    
    try:
        # Import serverless handler
        from src.uvip_ai.runpod_serverless import runpod_serverless
        
        # Call the handler
        result = runpod_serverless(payload)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Serverless handler error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Serverless processing failed: {str(e)}",
                "success": False
            }
        )


# Also support explicit /invoke path
@app.post("/invoke")
async def serverless_invoke(payload: dict = None):
    """Alternative endpoint path for Invoke."""
    if payload is None:
        return JSONResponse(status_code=400, content={"error": "No payload"})
    
    from src.uvip_ai.runpod_serverless import runpod_serverless
    result = runpod_serverless(payload)
    
    return JSONResponse(content=result)
