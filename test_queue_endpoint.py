#!/usr/bin/env python3
"""Test UVIP-AI Queue Serverless Endpoint"""
import requests
import json

ENDPOINT_ID = "46ifkgzb20xva0"
API_KEY = ""

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

headers = {
    "Content-Type": "application/json",
    "Authorization": API_KEY
}

payload = {
    "image": "",
    "latitude": -7.976,
    "longitude": 112.630
}

print(f"Testing Queue Serverless...")
print(f"URL: {url}")
print("Payload:", json.dumps(payload))
print("-"*70)

response = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\n✅ Status: {response.status_code}")
print("\n📊 Response:")
print(json.dumps(response.json(), indent=2))
