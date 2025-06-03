#!/bin/bash
# Run this to initialize collections in the Docker environment

echo "🚀 Initializing collections system in PostgreSQL (Docker)..."
docker exec -it jarvis-web-1 python init_collections_postgres.py