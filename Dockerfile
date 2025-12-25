# FinRobot API - Full Featured with All Agents
# Uses PyTorch CPU-only to reduce image size from 2GB to ~300MB

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    wkhtmltopdf \
    poppler-utils \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch CPU-only FIRST (from pytorch.org - smaller than CUDA version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_server.py .
COPY finrobot/ ./finrobot/ 

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
