"""
Heuristic pre-OCR image-quality signals for a KTP upload.

These are diagnostic, not a filter - the KTP pipeline still attempts full
extraction on every upload regardless of what this reports. The numbers
here explain, to a caller inspecting the response's `quality` block or a
human reviewing a low-confidence result, WHY a particular photo came back
thin (e.g. `resolution_ok: false` is a good hint that a low NIK confidence
isn't a parser bug), backing the response's `warnings` with concrete
signals rather than leaving the caller to guess from a wall of nulls.

None of this should be treated as an OCR-confidence proxy - a sharp,
well-lit, perfectly legible photo of the WRONG document (or of nothing at
all) would still score well here. See ktp/parser.py for the actual
field-level confidence, which is what should drive "trust this value"
decisions.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas import QualityAssessment

# Variance-of-Laplacian above this (measured on the image actually handed
# to Tesseract, i.e. post-crop/post-upscale) is treated as "sharp enough".
# Chosen empirically against this service's bundled real-photo sample, not
# a universal constant - a different camera/JPEG-quality mix could shift
# where this line should sit; treat it as a starting point to tune against
# more real (test) photos, the same way LABEL_PATTERNS in ktp/parser.py
# needs tuning against real cards. A saturating transform
# (min(variance / _BLUR_SATURATION, 1.0)) turns the raw, unbounded
# variance into a comparable 0-1 score rather than exposing raw units that
# have no intrinsic meaning to a caller.
_BLUR_SATURATION = 800.0

# Narrower than this (after preprocess_for_ktp_ocr's own upscaling) and
# small printed text is usually unreadable no matter how sharp the photo
# is - matches preprocessing._OCR_TARGET_WIDTH, the width that module
# already upscales toward.
_MIN_OK_WIDTH = 900

# Pixel value at/above which a pixel is considered "near-saturated" -
# candidate glare, not just "bright".
_GLARE_PIXEL_THRESHOLD = 250


def _blur_score(gray: np.ndarray) -> float:
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(variance / _BLUR_SATURATION, 1.0))


def _glare_detected(gray: np.ndarray) -> bool:
    """Specular glare/reflection is a spatially CONCENTRATED bright spot,
    not just "a lot of bright pixels" - a plain white document background
    (or a page rendered on white, as this service's own synthetic test
    samples are) can legitimately be 90%+ near-saturated pixels without
    any actual glare on it at all. Looking at raw overexposed-pixel
    fraction alone (an earlier version of this function did) flagged
    exactly that false positive.

    Instead: find the largest CONNECTED near-saturated region and flag
    glare only when it covers a meaningful chunk of the frame (not just a
    few stray blown-out pixels) but clearly isn't the majority of the
    frame (which would mean "bright/white background", not "reflection
    hotspot"). The exact bounds are a heuristic starting point - see the
    module docstring - not a calibrated detector.
    """
    mask = (gray >= _GLARE_PIXEL_THRESHOLD).astype(np.uint8)
    total = mask.size
    overexposed_ratio = float(mask.sum()) / total
    if overexposed_ratio < 0.01 or overexposed_ratio > 0.6:
        return False
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return False
    largest_component_ratio = float(stats[1:, cv2.CC_STAT_AREA].max()) / total
    return 0.01 <= largest_component_ratio <= 0.35


def _lighting(gray: np.ndarray, glare: bool) -> bool:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    # Too dark, too bright, too flat (low contrast - a hallmark of a
    # hazy/washed-out or heavily backlit shot), or glare all count as
    # "not ok". These thresholds are intentionally loose - this is meant
    # to catch clearly bad lighting, not grade a merely mediocre photo.
    return 40 <= mean <= 235 and std > 20 and not glare


def assess_quality(
    original_bgr: np.ndarray,
    cropped_bgr: np.ndarray | None,
    ocr_ready_gray: np.ndarray,
) -> QualityAssessment:
    """original_bgr: the raw uploaded frame, before any processing - used
    only for document_area_ratio (how much of the original frame the
    detected card actually occupied).

    cropped_bgr: the output of preprocessing.detect_and_crop_card, or None
    when card detection didn't find anything to crop (a flat scan, or a
    photo where detection genuinely failed) - document_detected and
    perspective_corrected are both False in that case.

    ocr_ready_gray: the actual grayscale image about to be handed to
    Tesseract (post-crop, post-upscale, one representative preprocessing
    variant) - blur/lighting are measured HERE, not on the original raw
    upload, since this is what actually determines whether OCR can read
    the text.
    """
    document_detected = cropped_bgr is not None
    # This service's card detection is a single perspective-warp step
    # (preprocessing._detect_and_crop_card) - whenever it finds a card at
    # all, the crop IS already the perspective-corrected result, so these
    # two flags are equivalent given this pipeline. Kept as two separate
    # fields (rather than one) to match the requested response shape and
    # because a future detector that can crop without a full 4-point warp
    # (e.g. an axis-aligned bounding box for an already-flat scan) would
    # need to set them independently.
    perspective_corrected = document_detected

    blur = _blur_score(ocr_ready_gray)
    glare = _glare_detected(ocr_ready_gray)
    lighting_ok = _lighting(ocr_ready_gray, glare)
    resolution_ok = ocr_ready_gray.shape[1] >= _MIN_OK_WIDTH

    area_ratio = None
    if cropped_bgr is not None:
        orig_area = original_bgr.shape[0] * original_bgr.shape[1]
        crop_area = cropped_bgr.shape[0] * cropped_bgr.shape[1]
        if orig_area:
            area_ratio = round(crop_area / orig_area, 3)

    return QualityAssessment(
        document_detected=document_detected,
        perspective_corrected=perspective_corrected,
        blur_score=round(blur, 3),
        resolution_ok=resolution_ok,
        lighting_ok=lighting_ok,
        glare_detected=glare,
        document_area_ratio=area_ratio,
    )
