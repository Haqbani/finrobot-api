# FinRobot API Server Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    gunicorn \
    openai \
    ag2[openai]

# Copy application code
COPY . .

# Install finrobot
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
