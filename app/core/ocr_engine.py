"""Thin wrapper around pytesseract so the rest of the service never
imports it directly - if you swap engines later (PaddleOCR, a hosted
API) this is the only file that needs to change."""

from __future__ import annotations

import re

import numpy as np
import pytesseract
from pytesseract import Output

from app.config import settings

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def image_to_text(image: np.ndarray) -> str:
    return pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGES, config="--oem 3 --psm 6")


def image_to_data(image: np.ndarray) -> list[dict]:
    """Word-level OCR: same engine/config as image_to_text, but returns
    each recognized word's pixel bounding box (plus Tesseract's own
    block/paragraph/line grouping) instead of a flat string.

    Needed by signature_locator's scanned-document fallback, which has
    to know WHERE a signer's name sits on the page, not just that it's
    present somewhere in the text - image_to_text alone throws that
    position away.
    """
    data = pytesseract.image_to_data(
        image, lang=settings.OCR_LANGUAGES, config="--oem 3 --psm 6", output_type=Output.DICT
    )
    words: list[dict] = []
    for i in range(len(data["text"])):
        text = data["text"][i]
        if not text.strip():
            continue
        words.append(
            {
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "conf": data["conf"][i],
                "block_num": data["block_num"][i],
                "par_num": data["par_num"][i],
                "line_num": data["line_num"][i],
            }
        )
    return words


def image_to_digits(image: np.ndarray, psm: int = 7) -> tuple[str, float]:
    """A second, INDEPENDENT OCR read restricted to a digits-only
    character whitelist - used for a cropped value region (see
    field_ocr.locate_nik_value_crop), not the whole page.

    The point of the whitelist isn't just "clean up the output afterward"
    - it constrains what Tesseract's recognizer can output AT ALL, so a
    glyph that's ambiguous between e.g. "0"/"O" or "1"/"I"/"l" gets forced
    to its digit reading directly, rather than first being read as a
    letter and needing a lossy strip-non-digits pass (which used to just
    delete such letters outright, silently shortening a correctly-read 16
    digit NIK to 15 - see ktp_parser._extract_nik / reconcile_nik).

    lang="eng" regardless of settings.OCR_LANGUAGES: the whitelist already
    restricts output to 0-9, so the Indonesian language model buys nothing
    here and would only slow this (small, cropped) pass down.

    psm 7 ("treat the image as a single text line") fits a label-value
    crop; pass 8 ("single word") for a tighter crop if the line still
    contains extra whitespace-separated noise.

    Returns (digits_only_string, avg_confidence as 0-1). Returns ("", 0.0)
    for an empty/invalid crop or when Tesseract found nothing - callers
    should treat that as "no candidate from this pass", not an error.
    """
    if image is None or image.size == 0:
        return "", 0.0
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(image, lang="eng", config=config, output_type=Output.DICT)
    texts: list[str] = []
    confs: list[float] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        texts.append(text)
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
    digits = re.sub(r"\D", "", "".join(texts))
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return digits, avg_conf


def tesseract_version() -> str:
    return str(pytesseract.get_tesseract_version())


def available_languages() -> list[str]:
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return []
