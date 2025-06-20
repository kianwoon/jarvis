FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Docker CLI and Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    netcat-openbsd \
    curl \
    ca-certificates \
    gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm for MCP servers that use npx
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy and install local MCP server
# Note: MCP files need to be copied from host during build
# For now, we'll install dependencies and the package will be installed manually
RUN mkdir -p /tmp/mcp

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model if spacy is installed
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" || echo "Spacy not installed, skipping model download"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

# Keep as root for now since docker-compose.yml specifies user: root
# USER appuser

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 