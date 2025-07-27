#!/usr/bin/env python3
"""
Test script to simulate the upload and see if the generator executes
"""
import requests

# Create a simple test file
test_content = b"Test PDF content"

files = {'file': ('test.pdf', test_content, 'application/pdf')}
data = {'collection_name': 'product_documentation'}

print("🔥 Making upload request...")

# Make the request and read the stream
response = requests.post(
    'http://localhost:8000/api/v1/documents/upload_pdf_progress',
    files=files,
    data=data,
    stream=True
)

print(f"🔥 Response status: {response.status_code}")
print(f"🔥 Response headers: {dict(response.headers)}")

if response.status_code == 200:
    print("🔥 Reading stream...")
    for line in response.iter_lines(decode_unicode=True):
        if line:
            print(f"Received: {line}")
else:
    print(f"🔥 Error: {response.text}")