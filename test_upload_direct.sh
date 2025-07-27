#!/bin/bash

echo "Testing upload endpoint directly..."

curl -X POST "http://localhost:8000/api/v1/documents/upload_pdf_progress" \
  -H "accept: text/event-stream" \
  -F "file=@/Users/kianwoonwong/Downloads/jarvis/test_generator.py" \
  -F "collection_name=product_documentation" \
  --no-buffer