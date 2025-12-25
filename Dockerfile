# FinRobot API - Lightweight Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only essential requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API server
COPY api_server.py .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
