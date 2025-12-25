# FinRobot API - Full Featured with All Agents
# Uses pinned versions from working local environment

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch CPU-only first (from pytorch.org index)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install pinned requirements (no resolution needed - all versions specified)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_server.py .
COPY finrobot/ ./finrobot/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
