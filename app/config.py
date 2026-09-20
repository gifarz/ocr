"""
Configuration for the CompliFi OCR service.

Everything is read from environment variables so this service can be
deployed (via PM2, systemd, or Docker) completely independently of the
CompliFi Next.js/Express app - it knows nothing about CompliFi's database,
Kong gateway, or auth model. It only exposes an HTTP API guarded by a
single shared API key.

load_dotenv() below reads a .env file in the current working directory
(if one exists) into the process environment before Settings reads
anything - without this, a .env file sitting next to main.py does
nothing on its own; os.getenv() only ever sees real environment
variables. In production (PM2/systemd/Docker) you can set real env vars
instead and .env becomes optional - dotenv silently no-ops if the file
isn't there.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Shared-secret auth between the CompliFi backend and this service.
    # Generate with e.g. `openssl rand -hex 32`. Required in production;
    # if left empty, auth is disabled (useful for local dev only).
    API_KEY: str = os.getenv("OCR_SERVICE_API_KEY", "")

    # Tesseract language packs to use. "ind+eng" covers Indonesian KTPs
    # and documents that mix in English terms/numbers. Requires the
    # tesseract-ocr-ind apt package to be installed on the host.
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "ind+eng")

    # Optional explicit path to the tesseract binary (rarely needed on
    # Linux where it's on PATH after `apt-get install tesseract-ocr`).
    TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD") or None

    # Max upload size in bytes (default 15 MB) - reject anything larger
    # before it ever reaches image processing.
    MAX_UPLOAD_BYTES: int = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))

    # If true, log the raw OCR text for each request at DEBUG level.
    # Keep this off in production since KTP scans contain NIK/PII -
    # logging raw text defeats the "don't persist ID data" posture.
    DEBUG_LOG_RAW_TEXT: bool = os.getenv("OCR_DEBUG_LOG_RAW_TEXT", "false").lower() == "true"


settings = Settings()
