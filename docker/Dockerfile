FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY backend/ /app/backend/
COPY data/ /app/data/

ENV MODEL_DIR=/app/data/ner_x5_88 \
    MAX_LEN=128

EXPOSE 8080

# Позволяем задавать число воркеров через ENV WORKERS (дефолт 2)
CMD ["/bin/sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --workers ${WORKERS:-2}"]
