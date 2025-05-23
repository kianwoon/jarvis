FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY MCP/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY MCP/ .

# Expose the port
EXPOSE 9000

# Command to run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--reload"] 