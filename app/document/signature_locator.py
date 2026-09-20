"""
Locates where a named signer's signature specimen should be stamped on
a document: it finds their printed name inside a signature block (e.g.
the "Pemohon" column on CompliFi's Peruri-style forms) and computes the
coordinates of the blank cell directly above it.

Why this is a geometry problem, not an OCR problem, on real CompliFi
documents: a form like the Peruri "Pernyataan Kerahasiaan" template is
a digitally generated PDF (see pdf_text.has_extractable_text) - every
printed name and every table rule is already a positioned object in
the file. Confirmed by inspecting that exact template: the printed
name "Muhammad Gifar Z." sits at fixed PDF-point coordinates, and the
table around it is drawn from thin filled rectangles whose edges give
the exact column and row boundaries - no pixels involved. So this
module's primary path, `_locate_native`, reads that structure directly
instead of rasterizing the page and re-discovering it with Tesseract,
which would only add a lossy image-resolution round trip for
information the PDF already stores exactly.

`_locate_scanned` is the fallback for when a document genuinely has no
extractable text - a photographed/scanned PDF, or a bare image upload
- where the same structure has to be recovered from pixels instead via
OCR word boxes + morphological line detection. That path is inherently
less certain; its results are scored lower and, like every other field
in this service (see schemas.FieldResult's docstring), should be shown
to a human for confirmation rather than used to silently place a
signature.
"""

from __future__ import annotations

import difflib
import re

import cv2
import fitz  # PyMuPDF
import numpy as np

from app.core import ocr_engine
from app.core.pdf_text import RuleLine, TextLine, extract_lines, extract_rule_lines, has_extractable_text
from app.schemas import SignatureBoxResult

# How tall a signature box is allowed to be when only one bounding rule
# (not two) was found above the name - stops a false-positive search
# from producing a box the size of half the page.
MAX_BOX_HEIGHT = 160.0

# Minimum name-match score (0-1) to accept a line as "this is the
# signer's printed name" rather than a coincidental partial match.
MIN_MATCH_SCORE = 0.6


def _normalize(name: str) -> list[str]:
    name = re.sub(r"[.,]", " ", name.lower())
    return [tok for tok in name.split() if tok]


def _name_match_score(candidate_tokens: list[str], query_tokens: list[str]) -> float:
    """Scores how well a printed line (e.g. "Muhammad Gifar Z.") matches
    a full name supplied by the caller (e.g. "Muhammad Gifar Zaini").

    Signature blocks routinely abbreviate a middle/last name down to
    its initial. A plain string-similarity ratio scores that pair
    poorly even though it's the correct person, so each token pair
    counts as matching if the tokens are equal OR one is a single
    letter matching the other's first letter (periods are already
    stripped by _normalize, so "Z." arrives here as "z"). Token
    matching is blended with a whole-string similarity ratio as a
    sanity check, so two unrelated names sharing one initial don't
    outrank a genuinely close match.
    """
    if not candidate_tokens or not query_tokens:
        return 0.0
    matched = 0
    cursor = 0
    for qt in query_tokens:
        for j in range(cursor, len(candidate_tokens)):
            ct = candidate_tokens[j]
            if ct == qt or (len(ct) == 1 and ct == qt[0]) or (len(qt) == 1 and qt == ct[0]):
                matched += 1
                cursor = j + 1
                break
    token_score = matched / len(query_tokens)
    ratio = difflib.SequenceMatcher(None, " ".join(candidate_tokens), " ".join(query_tokens)).ratio()
    return 0.7 * token_score + 0.3 * ratio


def _candidate_lines(lines: list[TextLine], signer_name: str) -> list[tuple[TextLine, float]]:
    """All lines scoring above threshold, best first.

    A signer's full name often appears MORE than once in a document -
    e.g. once as an identity field ("Nama: Muhammad Gifar Zaini" in a
    header table) and once, possibly abbreviated, under the actual
    signature line ("Muhammad Gifar Z."). Text similarity alone favors
    the unabbreviated header hit even though it's the wrong one for
    signature placement, so this returns every plausible hit and lets
    the caller pick using geometry quality (see `_locate_native`), not
    just the highest text score.
    """
    query_tokens = _normalize(signer_name)
    scored = [(line, _name_match_score(_normalize(line.text), query_tokens)) for line in lines]
    scored = [(line, score) for line, score in scored if score >= MIN_MATCH_SCORE]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _column_bounds(rules: list[RuleLine], name_rect: fitz.Rect) -> tuple[float, float, bool]:
    """Nearest vertical rules bracketing the name's x-range, restricted
    to rules that actually span the name's row - not just the nearest
    vertical rule anywhere on the page, which could belong to an
    unrelated table above or below the signature block. The trailing
    bool reports whether BOTH sides were found from real rules (a
    genuine table cell) rather than guessed."""
    x_mid = (name_rect.x0 + name_rect.x1) / 2
    left = right = None
    for r in rules:
        if r.orientation != "v":
            continue
        if not (r.start - 2 <= name_rect.y0 and r.end + 2 >= name_rect.y1):
            continue
        if r.pos <= name_rect.x0 and (left is None or r.pos > left):
            left = r.pos
        if r.pos >= name_rect.x1 and (right is None or r.pos < right):
            right = r.pos
    bounded = left is not None and right is not None
    if left is None:
        left = x_mid - 90.0  # generous fallback half-width if no column rule was found
    if right is None:
        right = x_mid + 90.0
    return left, right, bounded


def _row_bounds(rules: list[RuleLine], name_rect: fitz.Rect, column: tuple[float, float]) -> tuple[float, float, bool]:
    """The blank cell directly ABOVE the printed name: bounded below by
    the rule separating it from the name row, and above by the next
    rule up (the bottom of the header row above it) - rather than a
    fixed offset, so this adapts to whatever row height a given
    template uses. The trailing bool reports whether TWO enclosing
    rules were found (a genuine, fully-bounded cell) rather than one
    edge guessed from MAX_BOX_HEIGHT."""
    left, right = column
    x_mid = (left + right) / 2
    candidates = sorted(
        (
            r.pos
            for r in rules
            if r.orientation == "h" and r.start - 2 <= x_mid <= r.end + 2 and r.pos <= name_rect.y0 + 1
        ),
        reverse=True,  # nearest-above first
    )
    bounded = len(candidates) >= 2
    if bounded:
        bottom, top = candidates[0], candidates[1]
    elif len(candidates) == 1:
        bottom = candidates[0]
        top = max(bottom - MAX_BOX_HEIGHT, 0.0)
    else:
        bottom = name_rect.y0
        top = max(bottom - MAX_BOX_HEIGHT, 0.0)
    if bottom - top > MAX_BOX_HEIGHT:
        top = bottom - MAX_BOX_HEIGHT
    return top, bottom, bounded


def _pick_best_geometry(
    candidates: list[tuple[TextLine, float]], rules: list[RuleLine]
) -> tuple[TextLine, float, float, float, float, float, bool] | None:
    """Among every name-like line on the page, picks the one that sits
    in a genuinely ruled table cell over one that merely scores higher
    on text similarity - see `_candidate_lines`'s docstring for why a
    header identity field can outscore the real (possibly abbreviated)
    signature-block line otherwise. Falls back to the highest-scoring
    candidate, geometry-guessed, only if NONE of them sit in a fully
    ruled cell."""
    best_unbounded = None
    for line, score in candidates:
        col_left, col_right, col_bounded = _column_bounds(rules, line.rect)
        row_top, row_bottom, row_bounded = _row_bounds(rules, line.rect, (col_left, col_right))
        fully_bounded = col_bounded and row_bounded
        result = (line, score, col_left, row_top, col_right, row_bottom, fully_bounded)
        if fully_bounded:
            return result
        if best_unbounded is None:
            best_unbounded = result
    return best_unbounded


def _locate_native(doc: fitz.Document, signer_name: str) -> list[SignatureBoxResult]:
    results = []
    for page in doc:
        candidates = _candidate_lines(extract_lines(page), signer_name)
        if not candidates:
            continue
        rules = extract_rule_lines(page)
        picked = _pick_best_geometry(candidates, rules)
        if picked is None:
            continue
        line, score, col_left, row_top, col_right, row_bottom, fully_bounded = picked
        confidence = min(score, 1.0) if fully_bounded else min(score, 1.0) * 0.7
        results.append(
            SignatureBoxResult(
                page_index=page.number,
                matched_text=line.text.strip(),
                match_confidence=round(confidence, 2),
                x0=round(col_left, 1),
                y0=round(row_top, 1),
                x1=round(col_right, 1),
                y1=round(row_bottom, 1),
                page_width=round(page.rect.width, 1),
                page_height=round(page.rect.height, 1),
                method="pdf_native",
            )
        )
    return results


# ---------------------------------------------------------------------
# Scanned/photographed fallback - no native text to read from.
# ---------------------------------------------------------------------


def _bounding_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]


def _detect_pixel_rules(image: np.ndarray) -> list[RuleLine]:
    """Recovers horizontal/vertical table rules from pixels using
    morphological opening: erode the binarized page with a long, thin
    kernel so only long straight strokes survive (short text glyphs
    don't), then read off the surviving shapes' bounding boxes. This is
    the pixel equivalent of pdf_text.extract_rule_lines, needed because
    a scan has no vector objects to read directly."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = binary.shape
    rules: list[RuleLine] = []

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 20, 20), 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    for x, y, cw, ch in _bounding_boxes(h_mask):
        rules.append(RuleLine("h", y + ch / 2, x, x + cw))

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 20, 20)))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    for x, y, cw, ch in _bounding_boxes(v_mask):
        rules.append(RuleLine("v", x + cw / 2, y, y + ch))

    return rules


def _locate_scanned(images: list[np.ndarray], signer_name: str) -> list[SignatureBoxResult]:
    query_tokens = _normalize(signer_name)
    results = []
    for page_index, image in enumerate(images):
        words = ocr_engine.image_to_data(image)
        grouped: dict[tuple[int, int, int], list] = {}
        for w in words:
            grouped.setdefault((w["block_num"], w["par_num"], w["line_num"]), []).append(w)

        candidates: list[tuple[TextLine, float]] = []
        for items in grouped.values():
            items.sort(key=lambda it: it["left"])
            text = " ".join(it["text"] for it in items if it["text"].strip())
            if not text:
                continue
            score = _name_match_score(_normalize(text), query_tokens)
            if score < MIN_MATCH_SCORE:
                continue
            x0 = min(it["left"] for it in items)
            y0 = min(it["top"] for it in items)
            x1 = max(it["left"] + it["width"] for it in items)
            y1 = max(it["top"] + it["height"] for it in items)
            candidates.append((TextLine(page_index=page_index, text=text, rect=fitz.Rect(x0, y0, x1, y1)), score))
        candidates.sort(key=lambda pair: pair[1], reverse=True)

        rules = _detect_pixel_rules(image)
        picked = _pick_best_geometry(candidates, rules)
        if picked is None:
            continue
        line, score, col_left, row_top, col_right, row_bottom, fully_bounded = picked
        confidence = min(score, 1.0) * (0.85 if fully_bounded else 0.6)  # pixel path is inherently less certain
        h, w = image.shape[:2]
        results.append(
            SignatureBoxResult(
                page_index=page_index,
                matched_text=line.text.strip(),
                match_confidence=round(confidence, 2),
                x0=round(col_left, 1),
                y0=round(row_top, 1),
                x1=round(col_right, 1),
                y1=round(row_bottom, 1),
                page_width=float(w),
                page_height=float(h),
                method="ocr_scanned",
            )
        )
    return results


def locate_signature_box(
    file_bytes: bytes,
    content_type: str,
    signer_name: str,
    rasterized_pages: list[np.ndarray] | None = None,
) -> list[SignatureBoxResult]:
    """Entry point: returns one SignatureBoxResult per page where
    `signer_name` was found in a signature block, each carrying the
    coordinates of the blank cell directly above their printed name.

    Coordinates are in the document's own space: PDF points (top-left
    origin) for the native-PDF path, pixel coordinates for the scanned
    fallback - see `method` on each result and `page_width`/
    `page_height` to convert as needed.

    `rasterized_pages`, if the caller already produced them (e.g. the
    existing OCR endpoints in main.py rasterize every upload anyway),
    is reused instead of rendering the file a second time.
    """
    if content_type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            if has_extractable_text(doc):
                return _locate_native(doc, signer_name)
        finally:
            doc.close()
        # No extractable text - a scanned PDF. Fall through to the pixel path.

    images = rasterized_pages
    if images is None:
        from app.core.preprocessing import load_pages  # local import: avoids a cycle at module load time

        images = load_pages(file_bytes, content_type)
    return _locate_scanned(images, signer_name)
