"""
Extracts candidate values for the two free-text fields on CompliFi's
e-Meterai stamp form that come from the document itself rather than the
signer's identity: `docdate` and `docvalue`.

This is intentionally a *candidate generator*, not a single confident
answer: a contract PDF can contain several dates (signing date, due
date, effective date) and several currency figures. We surface the most
likely pick plus the full candidate list so the CompliFi frontend can
offer a dropdown ("we found 3 dates - which is the document date?")
instead of silently guessing wrong.
"""

from __future__ import annotations

import re

from app.schemas import DocumentExtractionResult, FieldResult

MONTHS_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}

# "12 Januari 2026", "12 Jan 2026"
DATE_TEXTUAL_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(MONTHS_ID.keys()) + r")\w*\s+(\d{4})\b",
    flags=re.IGNORECASE,
)
# "12-01-2026", "12/01/2026", "2026-01-12"
DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2}[\-/]\d{1,2}[\-/]\d{4}|\d{4}[\-/]\d{1,2}[\-/]\d{1,2})\b")

# "Rp 50.000.000", "Rp50,000,000", "IDR 50.000.000,00"
CURRENCY_RE = re.compile(
    r"\b(?:Rp\.?|IDR)\s?([\d.,]{4,})\b",
    flags=re.IGNORECASE,
)

# Lines that label the document's own value/date, boosting confidence
# when a candidate sits on or right after one of these.
DOCVALUE_LABEL_RE = re.compile(r"nilai\s*(dokumen|transaksi|kontrak)?", flags=re.IGNORECASE)
DOCDATE_LABEL_RE = re.compile(r"tanggal\s*(dokumen|kontrak|perjanjian)?", flags=re.IGNORECASE)


def _normalize_textual_date(day: str, month_name: str, year: str) -> str:
    month = MONTHS_ID.get(month_name.lower(), 0)
    return f"{int(day):02d}-{month:02d}-{year}"


def _find_dates(raw_text: str) -> list[tuple[str, str]]:
    """Returns (normalized_value, original_matched_text) pairs.

    Keeping the original matched substring alongside the normalized
    value matters for _best_pick below: a textual date like "12 Januari
    2026" normalizes to "12-01-2026", which never appears verbatim in
    the source line, so label-line matching has to search for the
    original text, not the normalized one.
    """
    found: list[tuple[str, str]] = []
    for m in DATE_TEXTUAL_RE.finditer(raw_text):
        found.append((_normalize_textual_date(m.group(1), m.group(2), m.group(3)), m.group(0)))
    for m in DATE_NUMERIC_RE.finditer(raw_text):
        found.append((m.group(1), m.group(1)))
    # De-dupe on normalized value while preserving first-seen order.
    seen: set[str] = set()
    ordered = []
    for normalized, original in found:
        if normalized not in seen:
            seen.add(normalized)
            ordered.append((normalized, original))
    return ordered


def _find_values(raw_text: str) -> list[tuple[str, str]]:
    found = [m.group(1).rstrip(".,") for m in CURRENCY_RE.finditer(raw_text)]
    seen: set[str] = set()
    ordered = []
    for v in found:
        if v not in seen:
            seen.add(v)
            ordered.append((v, v))  # currency values match their own text verbatim
    return ordered


def _best_pick(raw_text: str, candidates: list[tuple[str, str]], label_re: re.Pattern) -> FieldResult:
    if not candidates:
        return FieldResult()

    # Prefer a candidate whose ORIGINAL matched text appears on the same
    # line as a recognizable label ("Nilai Dokumen: Rp 50.000.000") over
    # the first match in the document, since contracts are full of
    # unrelated dates/amounts. Matching on the original text (not the
    # normalized value) matters for textual dates - "12 Januari 2026"
    # normalizes to "12-01-2026", which never appears verbatim in the
    # source line.
    for line in raw_text.splitlines():
        if label_re.search(line):
            for normalized, original in candidates:
                if original in line:
                    return FieldResult(value=normalized, confidence=0.85, source="ocr")

    # No labeled line matched - fall back to the first candidate found,
    # but say so with a lower confidence score.
    return FieldResult(value=candidates[0][0], confidence=0.4, source="ocr")


def parse_document(raw_text: str) -> DocumentExtractionResult:
    dates = _find_dates(raw_text)
    values = _find_values(raw_text)

    return DocumentExtractionResult(
        docdate=_best_pick(raw_text, dates, DOCDATE_LABEL_RE),
        docvalue=_best_pick(raw_text, values, DOCVALUE_LABEL_RE),
        docdate_candidates=[normalized for normalized, _original in dates],
        docvalue_candidates=[normalized for normalized, _original in values],
    )
