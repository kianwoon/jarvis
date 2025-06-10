#!/bin/bash
# Run the app locally with M3 GPU support

echo "Starting Jarvis locally with Apple M3 GPU support..."

# Load local environment variables
export $(cat .env.local | grep -v '^#' | xargs)

# Run the app
echo "Environment: ENABLE_QWEN_RERANKER=$ENABLE_QWEN_RERANKER"
echo "Device: QWEN_RERANKER_DEVICE=$QWEN_RERANKER_DEVICE"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000