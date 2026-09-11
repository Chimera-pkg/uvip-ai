#!/usr/bin/env python3
"""
Test script for UVIP-AI Serverless on RunPod
Copy this to your local computer and run after generating API key from dashboard
"""
import requests
import json
from pathlib import Path

# CONFIGURATION - REPLACE THESE VALUES!
ENDPOINT_ID = "46ifkgzb20xva0"  # From your dashboard (this part only!)
API_KEY = "YOUR_API_KEY_HERE"   # Generate from dashboard → Quick start → Generate key

ENDPOINT_URL = f"https://{ENDPOINT_ID}.rp.runpod.net"
print(f" Endpoint URL: {ENDPOINT_URL}")
print("=" * 70)


def test_health():
    """Test health endpoint (may require authentication)"""
    print("\n Testing health check...")
    try:
        response = requests.get(f"{ENDPOINT_URL}/health", timeout=15)
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"   Response: {result}")
            return True
        else:
            print(f"   Warning: {result}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_inference_simple(image_path: str):
    """Simple test with minimal image data"""
    print(f"\n📸 Testing with simple payload (no real image)...")
    
    payload = {
        "image": "",  # Empty/invalid image will trigger error but shows pipeline works
        "latitude": -7.976,
        "longitude": 112.630,
        "filename": "test_empty_image.jpg"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"ApiKey {API_KEY}"
    }
    
    print("⏳ Sending request (first request may take 10-30s for cold start)...")
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/serverless/invoke",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"\n✅ Response received!")
        print(f"Status Code: {response.status_code}")
        print(f"\n📊 Full Response:")
        print(json.dumps(response.json(), indent=2))
        
        return response.json()
        
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (cold start too slow)")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def test_with_real_image(image_path: str):
    """Test with actual image file"""
    print(f"\n📸 Testing with real image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Read image as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Convert to hex string (required by RunPod serverless)
        image_hex = image_bytes.hex()
        
        payload = {
            "image": image_hex,
            "latitude": -7.976,
            "longitude": 112.630,
            "filename": Path(image_path).name
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"ApiKey {API_KEY}"
        }
        
        print("⏳ Sending request... This may take 10-30 seconds (cold start)")
        
        response = requests.post(
            f"{ENDPOINT_URL}/serverless/invoke",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"\n✅ Response received!")
        print(f"Status Code: {response.status_code}")
        print(f"\n📊 Results:")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        return result
        
    except FileNotFoundError:
        print(f"❌ Image not found: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("UVIP-AI Serverless Test Suite")
    print("=" * 70)
    
    # Check if API key is configured
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  WARNING: You need to generate an API key first!")
        print("\nTo generate API key:")
        print("1. Go to RunPod dashboard → Your endpoint 'uvip-ai'")
        print("2. Look for 'Quick start' section")
        print("3. Click 'Generate key' button")
        print("4. Copy the generated key")
        print("5. Replace 'YOUR_API_KEY_HERE' above with your actual key")
        print("\nSkipping tests until API key is provided...")
        input("\nPress Enter when you've updated the API key...")
    else:
        # Run tests
        test_health()
        print("\n" + "=" * 70)
        test_inference_simple(None)
        
        # Uncomment this line if you have a real image to test
        # test_with_real_image("data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg")
