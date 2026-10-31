FROM python:3.13-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by LightGBM; curl: container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/cwru/ ./data/cwru/
COPY .streamlit/ ./.streamlit/
COPY ingest.py ./

# Train at build time so the model artifact always matches the library versions in this image
# (fails the build loudly if the CWRU files are missing or unusable).
RUN python src/train.py

ENV PYTHONUNBUFFERED=1

# Default command (overridden per service in docker-compose.yml)
CMD ["python", "src/inference_service.py"]
