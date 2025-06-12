#!/usr/bin/env python3
import requests

# Test the most basic endpoint
response = requests.get("http://localhost:8000/docs")
print(f"API Docs status: {response.status_code}")

# Test LLM endpoint directly
print("\nTesting LLM endpoint directly:")
response = requests.post(
    "http://localhost:8000/api/v1/generate_stream",
    json={"prompt": "Hi", "temperature": 0.7, "top_p": 1.0, "max_tokens": 10},
    stream=True,
    timeout=10
)

print(f"LLM Status: {response.status_code}")
for i, line in enumerate(response.iter_lines()):
    if i < 5:  # Show first 5 lines
        print(f"  Line {i}: {line}")
    else:
        print("  ... (truncated)")
        break