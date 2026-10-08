FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY ingest.py simulator.py ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/inference_service.py"]
