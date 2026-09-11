#!/usr/bin/env python3
"""🚀 QUICK TEST SCRIPT - RunPod Serverless UVIP-AI"""
import requests
import json

# CONFIGURATION (EDIT THIS!)
ENDPOINT_ID = "46ifkgzb20xva0"
API_KEY = "PASTE_YOUR_API_KEY_HERE"  # Get from dashboard → Quick start → Generate key

url = f"https://{ENDPOINT_ID}.rp.runpod.net/serverless/invoke"
headers = {"Content-Type": "application/json", "Authorization": f"ApiKey {API_KEY}"}

print("="*70)
print("UVIP-AI Serverless Test")
print(f"Endpoint: {url}")
print("="*70)

# Simple test payload (empty image will return error but proves endpoint works)
payload = {
    "image": "",
    "latitude": -7.976,
    "longitude": 112.630
}

print("\n⏳ Sending request (first request: 10-30s cold start)...")
response = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\n✅ Status: {response.status_code}")
print("\n📊 Response:")
print(json.dumps(response.json(), indent=2))
print("="*70)
