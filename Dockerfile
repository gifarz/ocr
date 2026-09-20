FROM python:3.12-slim

# tesseract-ocr-ind pulls in the Indonesian trained-data language pack;
# tesseract-ocr is the base engine + eng data.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-ind libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8088

# Lets `docker compose` / any container orchestrator detect a stuck or
# crashed process and restart it automatically (see docker-compose.yml's
# restart: unless-stopped) rather than silently serving nothing. Uses
# Python's own stdlib instead of installing curl just for this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
