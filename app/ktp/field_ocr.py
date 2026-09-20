"""
Region-based, field-specific OCR for KTP fields.

The whole-page OCR + line-based text parsing in ktp/parser.py is fast and
covers most fields well, but for a field whose value shape is strictly
constrained and security-critical (NIK: exactly 16 digits), running OCR a
SECOND time - restricted to just the pixels immediately to the right of
the label, and restricted to digits only via ocr_engine.image_to_digits'
character whitelist - gives an independent read that's far less prone to
a digit being misread as a similar-looking letter in the first place,
since letters simply aren't in that pass's output alphabet at all.

This is a cross-check against the whole-text tiered extraction in
ktp_parser._extract_nik, not a replacement for it - see
ktp_parser.reconcile_nik for how the two are combined.
"""

from __future__ import annotations

import re

import numpy as np

# Vertical padding (as a fraction of the label word's own line height)
# added above/below the cropped value region - Tesseract's own line-height
# estimate tends to run a touch tight around a real font's ascenders and
# descenders, so a hairline pad avoids clipping the value text right at
# its top/bottom edge.
_VERTICAL_PAD_RATIO = 0.35

_NIK_LABEL_RE = re.compile(r"^NIK$", re.IGNORECASE)


def locate_nik_value_crop(image: np.ndarray, words: list[dict]) -> np.ndarray | None:
    """Finds a word reading exactly "NIK" among `words` (as returned by
    ocr_engine.image_to_data FOR THIS SAME image - coordinates are only
    meaningful against the exact image they came from) and returns a crop
    of the value region immediately to its right, on the same OCR-detected
    line.

    Deliberately requires an EXACT "NIK" token (after stripping
    surrounding punctuation), not a prefix/substring match: if Tesseract's
    own tokenizer glued the label and a misread colon straight onto the
    digits into one word (e.g. "NIK231710..."), there's no reliable way to
    know from the bounding box alone where the label ends and the value
    begins, and guessing would risk feeding the digit-only pass a crop
    that still starts mid-label - better to skip this variant's region
    candidate entirely than risk a systematically-wrong crop; the
    whole-text tiered path (ktp_parser._extract_nik) already has its own,
    separately-tuned handling for exactly this glued-token case.

    Returns None whenever no such line is found - callers should treat
    that as "no region-based candidate from this variant/preprocessing
    pass", not an error; a sufficiently degraded variant may simply not
    have a legible "NIK" label at all.
    """
    label_words = [w for w in words if _NIK_LABEL_RE.match(w["text"].strip(" :.-"))]
    if not label_words:
        return None
    # Several words on the page reading exactly "NIK" is unlikely on a
    # real single-card upload - take the first.
    label = label_words[0]

    same_line = [
        w
        for w in words
        if w["block_num"] == label["block_num"]
        and w["par_num"] == label["par_num"]
        and w["line_num"] == label["line_num"]
    ]
    right_edge_of_label = label["left"] + label["width"]
    value_words = [w for w in same_line if w["left"] >= right_edge_of_label - 5 and w is not label]
    if not value_words:
        # Label found, but Tesseract produced no separate token after it
        # on this line (e.g. the value merged into the same token as a
        # misread colon after all) - no safe crop to take, see docstring.
        return None

    h_img, w_img = image.shape[:2]
    line_height = label["height"]
    pad = max(2, int(line_height * _VERTICAL_PAD_RATIO))
    y0 = max(0, label["top"] - pad)
    y1 = min(h_img, label["top"] + line_height + pad)
    x0 = max(0, right_edge_of_label)
    x1 = min(w_img, max(w["left"] + w["width"] for w in value_words) + pad)

    if x1 - x0 < 10 or y1 - y0 < 5:
        return None
    return image[y0:y1, x0:x1]
