import requests
import json

# Replace this with your actual endpoint
API_URL = "https://64ptlsypgldn3v-4000.proxy.runpod.net/generate"  # or "http://127.0.0.1:8000/generate" if running without Apache


# Prompt to test
payload = {
    "prompt": "Write a Python function to reverse a string."
}

headers = {
    "Content-Type": "application/json"
}

# Send POST request
response = requests.post(API_URL, headers=headers, json=payload)

# Parse response
if response.status_code == 200:
    data = response.json()
    print("🧠 AI Output:\n")
    print(data["output"])
else:
    print(f"❌ Request failed with status {response.status_code}")
    print(response.text)
