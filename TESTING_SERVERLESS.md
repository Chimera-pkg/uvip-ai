# 🧪 Testing RunPod Serverless UVIP-AI

## 📊 Current Status (From Your Dashboard)

```
✅ Endpoint: uvip-ai (Queue based)
⚡ Running: $0.00016/s
👷 Workers: 2 running
🔄 Rollout: 43% complete (in progress)
```

---

## ✅ METHOD 1: Test via "Bulk Requests" Tab (Browser)

### Step 1: Generate API Key

1. Go to **Quick start** section on dashboard
2. Click **"Generate key"**
3. Copy the API key shown

### Step 2: Make Test Request

In the cURL example dropdown, select `POST /runsync`:

```bash
curl https://<YOUR_ENDPOINT_ID>.rp.runpod.net/runsync \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: ApiKey <YOUR_API_KEY>" \
  -d '{
    "image": "",
    "latitude": -7.976,
    "longitude": 112.630,
    "filename": "test.jpg"
  }'
```

**Replace `<YOUR_ENDPOINT_ID>`** with your actual endpoint ID from dashboard.

---

## ✅ METHOD 2: Direct Browser Test (No CLI)

### Step 1: Get Public URL

From dashboard, find:
```
Endpoint URL: https://xxxxx46ifkgzb20xva0.rp.runpod.ai
```

(Usually looks like: `https://<endpoint-id>.rp.runpod.ai`)

### Step 2: Add Header via Browser Console

1. Open Developer Tools (F12)
2. Go to **Network** tab
3. Enable **"Preserve log"**

### Step 3: Create Test Form

Create an HTML file locally:

```html
<!DOCTYPE html>
<html>
<head><title>UVIP-AI Serverless Test</title></head>
<body>
<h1>Test AI Inference</h1>

<input type="file" id="imageInput" accept="image/*">
<br><br>
<label>Latitude:</label> <input type="number" id="lat" value="-7.976" step="0.001"><br>
<label>Longitude:</label> <input type="number" id="lng" value="112.630" step="0.001"><br><br>

<button onclick="testServerless()">Send Test Request</button>
<div id="result"></div>

<script>
async function testServerless() {
  const fileInput = document.getElementById('imageInput');
  const lat = document.getElementById('lat').value;
  const lng = document.getElementById('lng').value;
  
  if (!fileInput.files[0]) {
    alert('Please select an image first!');
    return;
  }
  
  // Read file as binary
  const arrayBuffer = await fileInput.files[0].arrayBuffer();
  const uint8Array = new Uint8Array(arrayBuffer);
  const binaryString = uint8Array.reduce((acc, byte) => acc + String.fromCharCode(byte), '');
  
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '⏳ Processing...';
  
  try {
    const response = await fetch('https://<YOUR-ENDPOINT-ID>.rp.runpod.ai/serverless/invoke', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: binaryString,
        latitude: parseFloat(lat),
        longitude: parseFloat(lng),
        filename: fileInput.files[0].name
      })
    });
    
    const data = await response.json();
    resultDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
    
    console.log('Serverless Response:', data);
  } catch (error) {
    resultDiv.innerHTML = '❌ Error: ' + error.message;
    console.error('Error:', error);
  }
}
</script>
</body>
</html>
```

4. Replace `<YOUR-ENDPOINT-ID>` with your actual ID
5. Open this HTML file in browser
6. Select image → Click "Send Test Request"

---

## ✅ METHOD 3: Using Python Script

Save as `test_serverless.py`:

```python
#!/usr/bin/env python3
"""Test script for RunPod Serverless UVIP-AI endpoint"""
import requests
import json
import base64
from pathlib import Path

# Configuration
ENDPOINT_URL = "https://xxxxx46ifkgzb20xva0.rp.runpod.ai"  # Replace with YOUR endpoint
API_KEY = "YOUR_API_KEY_HERE"  # Generate from dashboard → Quick start

def test_health():
    """Test health endpoint (optional, may require auth)"""
    print("Testing health check...")
    try:
        response = requests.get(f"{ENDPOINT_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Health check failed (may need auth): {e}")

def test_inference(image_path: str, latitude: float = -7.976, longitude: float = 112.630):
    """Test full AI inference pipeline"""
    print(f"\n📸 Testing with image: {image_path}")
    
    # Read image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Convert to hex string (RunPod expects hex-encoded binary)
    image_hex = image_bytes.hex()
    
    payload = {
        "image": image_hex,
        "latitude": latitude,
        "longitude": longitude,
        "filename": Path(image_path).name
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Send request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"ApiKey {API_KEY}"
    }
    
    print("\n⏳ Sending request... This may take 10-30 seconds (cold start)")
    response = requests.post(
        f"{ENDPOINT_URL}/serverless/invoke",
        headers=headers,
        json=payload,
        timeout=120  # 2 minutes timeout for first request
    )
    
    print(f"\n✅ Response received!")
    print(f"Status Code: {response.status_code}")
    print(f"\n📊 Results:")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()

if __name__ == "__main__":
    test_health()
    
    # Test with sample image (provide your own path!)
    test_image = "data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg"
    
    try:
        test_inference(test_image)
    except FileNotFoundError:
        print(f"❌ Image not found: {test_image}")
        print("   Provide a test image path or comment out this line")
