"""
Turns an uploaded file (JPEG/PNG photo of a KTP, or a PDF document) into
one or more clean grayscale numpy images ready for Tesseract.

Kept dependency-light on purpose: PDF pages are rasterized with PyMuPDF
(pure Python wheel, no poppler/system binary needed) rather than
pdf2image, since this service is meant to be easy to drop onto any VPS
next to the existing CompliFi deployment.
"""

from __future__ import annotations

import io

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
SUPPORTED_PDF_TYPE = "application/pdf"

# Render PDFs at ~220 DPI - enough for small printed text without
# ballooning processing time on a VPS.
PDF_RENDER_ZOOM = 220 / 72


def load_pages(file_bytes: bytes, content_type: str) -> list[np.ndarray]:
    """Returns a list of BGR numpy images, one per page (images -> 1 page)."""
    if content_type == SUPPORTED_PDF_TYPE:
        return _pdf_to_images(file_bytes)
    if content_type in SUPPORTED_IMAGE_TYPES:
        return [_bytes_to_image(file_bytes)]
    raise ValueError(
        f"Unsupported content type '{content_type}'. Expected a PDF or an image "
        f"({', '.join(sorted(SUPPORTED_IMAGE_TYPES))})."
    )


def _bytes_to_image(file_bytes: bytes) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _pdf_to_images(file_bytes: bytes) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
    return images


def preprocess_for_ocr(image: np.ndarray, force_threshold: bool = False, detect_card: bool = False) -> np.ndarray:
    """Grayscale + deskew + light denoise, ready for Tesseract - plus,
    when detect_card=True, a card-detect/crop/upscale pass first (see
    _detect_and_crop_card).

    detect_card defaults to False because this function is shared by
    every OCR-based endpoint, not just KTP extraction: a generic
    contract/invoice document (/v1/extract/document) is exactly the
    kind of mostly-blank, sparse-text page _detect_and_crop_card's
    "find the largest card-shaped quadrilateral" heuristic was never
    meant to run on, and testing confirmed it shouldn't - it found a
    spurious quadrilateral in a plain document photo and cropped out
    real content, regressing docvalue extraction. Only main.py's KTP
    endpoint should pass detect_card=True.

    Deliberately NOT running a blanket adaptive threshold by default.
    Earlier versions did, on the assumption that binarizing helps -
    but testing against a real KTP photo showed the opposite: on a
    clean, high-contrast card (whether a physical scan or a digital
    template), forced thresholding introduced artifacts that weren't
    there in plain grayscale - most damagingly, it turned the ":" after
    "NIK" into something Tesseract read as a literal "2", corrupting
    the identity number. Tesseract already does its own internal
    binarization, which handled this image correctly on its own.

    Pass force_threshold=True for the genuinely hard case this WAS
    meant for - a real phone photo with uneven lighting/shadow across
    the card - if you hit one where plain grayscale is unreadable.
    """
    if detect_card:
        image = _detect_and_crop_card(image)
        image = _upscale_if_small(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    if force_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
    return gray


# A KTP crop this narrow doesn't give Tesseract enough pixels per
# character to read reliably - upscale toward this target width before
# OCR. Chosen empirically against a real (blurry, phone-camera) photo
# where a ~590px-wide card crop OCR'd as near-total garbage but the
# same crop upscaled to ~1470px (2.5x) came back mostly readable.
_OCR_TARGET_WIDTH = 1400


def _upscale_if_small(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w >= _OCR_TARGET_WIDTH or w == 0:
        return image
    scale = min(_OCR_TARGET_WIDTH / w, 3.0)  # cap the scale - beyond 3x just blows up noise
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as (top-left, top-right, bottom-right, bottom-left) -
    the order cv2.getPerspectiveTransform needs, regardless of what
    order the contour's corners came back in."""
    rect = np.zeros((4, 2), dtype="float32")
    total = pts.sum(axis=1)
    rect[0] = pts[np.argmin(total)]
    rect[2] = pts[np.argmax(total)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    max_width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


# A KTP is a standard ID-1 card, 85.6mm x 54mm - long side / short side
# ~= 1.585. Some slack is needed since a hand-held photo's perspective
# and the corner-approximation below both distort the measured ratio a
# little, but this still rules out things like a doorframe or a sheet
# of paper in the background that Canny might otherwise pick up.
_KTP_ASPECT_RATIO = 85.6 / 54.0
_KTP_ASPECT_TOLERANCE = 0.5

# How many degrees a corner's interior angle may deviate from a true 90
# degrees before the quadrilateral is rejected as "not actually
# rectangular" - see _corner_angle_deviations for why this check exists
# at all: aspect ratio and area alone don't catch a badly-warped
# candidate whose SHAPE isn't a card's, because a thin, sheared
# quadrilateral can still happen to land at a card-like aspect ratio and
# a plausible area purely by coincidence. Calibrated against this
# service's two real bundled photos: the genuine card corner in
# sample_ktp_held_photo.jpg deviates up to ~21 degrees from a real
# perspective angle; a wrong quadrilateral found on a studio-clean, flat
# KTP template photo (whose true low-contrast outer edge Canny failed to
# close into a contour at all, so the largest contour it DID find was an
# internal watermark graphic instead) deviated up to ~34 degrees. 25 sits
# between the two - a starting point, not a proven-universal cutoff (see
# "Known limitations" in the README).
_MAX_CORNER_ANGLE_DEVIATION = 25.0

# How many of the largest external contours to try, in descending area
# order, before giving up - the true card boundary isn't always the
# single largest contour Canny finds (a higher-contrast internal
# graphic, logo, or watermark can outscore a card's own soft, low-
# contrast edge against a similarly-toned background), so rather than
# committing to "largest wins" and letting a wrong one slip through
# whenever it happens to also clear the area/aspect-ratio/rectangularity
# checks, several candidates are tried and the first (i.e. largest) one
# that passes ALL of them is used.
_MAX_CANDIDATE_CONTOURS = 5


def _corner_angle_deviations(rect: np.ndarray) -> list[float]:
    """rect: 4 points ordered (top-left, top-right, bottom-right,
    bottom-left), as returned by _order_points. Returns, for each
    corner, how many degrees its interior angle deviates from a true 90
    - a real card's corners stay close to 90 even under a plausible
    photo angle; a contour that isn't actually the card's outline (see
    _MAX_CORNER_ANGLE_DEVIATION) tends to have at least one corner far
    off from that, since nothing constrains an arbitrary 4-point
    approximation to be even roughly rectangular otherwise.
    """
    pts = [rect[i] for i in range(4)]
    deviations = []
    for i in range(4):
        prev_pt = pts[i - 1]
        cur_pt = pts[i]
        next_pt = pts[(i + 1) % 4]
        v1 = prev_pt - cur_pt
        v2 = next_pt - cur_pt
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            deviations.append(90.0)  # degenerate corner - treat as maximally wrong, not a divide-by-zero crash
            continue
        cos_angle = float(np.dot(v1, v2) / denom)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        deviations.append(abs(angle - 90.0))
    return deviations


def detect_and_crop_card(image: np.ndarray) -> np.ndarray:
    """Public wrapper around _detect_and_crop_card, for callers that need
    the crop result itself rather than the full multi-variant KTP
    preprocessing pipeline - currently just image_quality.assess_quality,
    called once per upload from ktp/pipeline.py to know whether a card was
    actually found (see that module for how "no crop happened" is
    detected: cropped.shape == image.shape)."""
    return _detect_and_crop_card(image)


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """Simplifies a contour down to a 4-point convex polygon, or returns
    None if it can't be (see the eps_mult comment below)."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    # Walk the simplification tolerance up until exactly 4 corners
    # survive - a single fixed epsilon is either too tight (leaves
    # extra corners from a slightly wavy detected edge) or too loose
    # (collapses real corners) depending on the photo, so this adapts
    # per-image instead of assuming one tolerance fits every case.
    for eps_mult in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1):
        candidate = cv2.approxPolyDP(hull, eps_mult * peri, True)
        if len(candidate) == 4:
            if not cv2.isContourConvex(candidate):
                return None
            return candidate.reshape(4, 2).astype("float32")
    return None


def _validated_card_quad(pts: np.ndarray, image_area: float) -> np.ndarray | None:
    """Runs every shape check a card candidate has to pass (area share,
    aspect ratio, AND rectangularity - see _MAX_CORNER_ANGLE_DEVIATION)
    and returns the ordered rect if it clears all of them, else None."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2
    height = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2
    if height == 0 or width == 0:
        return None
    ratio = max(width, height) / min(width, height)
    if abs(ratio - _KTP_ASPECT_RATIO) > _KTP_ASPECT_TOLERANCE:
        return None
    if max(_corner_angle_deviations(rect)) > _MAX_CORNER_ANGLE_DEVIATION:
        return None
    return rect


def _detect_and_crop_card(image: np.ndarray) -> np.ndarray:
    """Finds the largest card-shaped quadrilateral in the photo and
    perspective-warps it into a clean, cropped, top-down view - the
    difference between a flat scan/crop of just the KTP (most of this
    service's testing so far) and a photo where a much smaller card is
    held up in someone's hand against a room/background (a very common
    real-world submission for identity verification). Skipping this
    step on the latter feeds Tesseract - and this function's own
    deskew step, whose angle estimate is computed from ALL non-
    background pixels in the frame - a tiny sliver of actual card text
    swamped by a person's face, clothes, and the room behind them,
    which is what was producing near-total OCR garbage before this was
    added (verified against a real "student holding up their KTP"
    press photo).

    A bilateral filter (rather than a plain Gaussian blur) is used
    before edge detection specifically because it smooths out
    background texture - skin, fabric, a painted wall - while keeping
    the card's own edges sharp; a plain blur softens both equally and
    was losing the card boundary entirely in testing. The Canny
    thresholds are derived from the image's own median brightness
    (rather than fixed constants) so this isn't tuned to just one
    photo's lighting.

    Falls back to returning the image UNCHANGED whenever no
    sufficiently large, sufficiently card-shaped, sufficiently
    RECTANGULAR quadrilateral is found among the several largest
    contours - including when the input is already just the card (the
    common case for a flat scan), where forcing a crop would be more
    likely to clip real content than help.

    The rectangularity check specifically exists because of a real bug
    found against a clean, studio-style flat KTP template photo: its
    actual outer edge is a soft, low-contrast boundary against a
    similarly-toned plain background, which Canny + the morphological
    closing above never joined into one closed contour at all - so the
    largest contour actually found was an unrelated internal graphic (a
    watermark), which happened to ALSO clear the area and aspect-ratio
    checks below purely by coincidence, despite being a heavily sheared
    quadrilateral rather than anything resembling the card's true
    shape. That produced a perspective warp which sheared the card's
    text into visible nonsense (each label row ending up next to the
    WRONG value row) rather than failing safely - see
    test_detect_and_crop_card_rejects_a_non_rectangular_distractor for
    the regression test built from that exact photo, and "Known
    limitations" in the README for why a real fix (actually finding
    this photo's true card boundary) wasn't attempted here: rejecting a
    bad detection and falling back to the untouched image is far lower-
    risk than trying to widen contour detection enough to catch this
    case, which risks trading a rare bad-crop failure mode for a more
    common false-positive-crop one on other photos.
    """
    h, w = image.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    median = float(np.median(smoothed))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(smoothed, lower, upper)
    # Close small gaps in the card's boundary (a corner obscured by a
    # thumb, a reflection, a low-contrast stretch against a similarly-
    # toned background) into one connected outline before contour
    # detection, rather than several disconnected arcs.
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Try the several largest contours, in descending area order - not
    # just the single largest (see this function's own docstring for
    # why "largest wins, no further validation" let a wrong contour
    # through on a real test photo). The first candidate that clears
    # EVERY check (area share, exactly-4-corners, convex, aspect ratio,
    # rectangularity) is used.
    by_area = sorted(contours, key=cv2.contourArea, reverse=True)[:_MAX_CANDIDATE_CONTOURS]
    for contour in by_area:
        area = cv2.contourArea(contour)
        if area < 0.05 * image_area or area > 0.95 * image_area:
            continue
        pts = _approx_quad(contour)
        if pts is None:
            continue
        rect = _validated_card_quad(pts, image_area)
        if rect is not None:
            return _four_point_warp(image, pts)

    return image


MAX_AUTO_DESKEW_DEGREES = 15.0


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Corrects small rotations (a few degrees) from a hand-held photo.

    Uses the minAreaRect of all non-background pixels rather than a full
    Hough-line approach - cheaper and good enough for the +/-10deg tilts
    typical of phone photos of a card on a desk.

    IMPORTANT SAFETY CLAMP: minAreaRect over a sparse point cloud (a
    handful of short text lines scattered across an otherwise-white
    page, as in a typical PDF-rendered document rather than a dense
    KTP-card photo) can report a wildly wrong angle - e.g. 90 degrees -
    because the overall bounding shape of scattered lines doesn't
    reflect the actual per-line skew. A real photo skew is never that
    large, so anything beyond MAX_AUTO_DESKEW_DEGREES is treated as a
    bad estimate and skipped rather than applied, to avoid rotating an
    already-upright document into unreadable garbage. This was caught
    by testing against a synthetic document sample - see tests/.
    """
    inverted = cv2.bitwise_not(gray)
    coords = cv2.findNonZero(cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    if coords is None or len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3 or abs(angle) > MAX_AUTO_DESKEW_DEGREES:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

# ---------------------------------------------------------------------------
# Multi-variant preprocessing for KTP extraction
# ---------------------------------------------------------------------------
#
# A single "best-guess" preprocessing pipeline is not enough on a genuinely
# degraded source photo (a real "holding the card up" press photo, not a
# flat scan) - testing against one confirmed that different denoise
# strength/sharpening/upscale-factor choices each recover a DIFFERENT
# subset of fields cleanly, with little overlap between them, and no
# single variant recovering more than about half the card. Rather than
# committing to one setting that wins on average, preprocess_for_ktp_ocr
# below returns several variants for main.py to run OCR + parsing against
# independently; ktp_parser.merge_ktp_results then picks the best result
# PER FIELD across all of them. This is naturally more expensive
# (Tesseract runs once per variant) - acceptable for a single-page KTP
# photo, not something to reuse for the other, generally-larger-and-
# cleaner OCR endpoints.

_KTP_VARIANT_SPECS: list[tuple[float, str]] = [
    (3.0, "denoise7"),
    (3.0, "sharpen"),
    (2.5, "denoise7"),
    (4.0, "denoise7"),
    (3.5, "nodenoise"),
    (3.0, "deconv_defocus"),
    (3.0, "deconv_motion"),
]


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A line-shaped point-spread function simulating linear motion
    blur of the given length (px) and direction - the classic
    "camera shake" blur shape, as opposed to the roughly circular
    blur a lens produces when the subject is simply out of focus
    (see _defocus_blur_kernel)."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = kernel.sum()
    return kernel / total if total else kernel


def _defocus_blur_kernel(radius: int) -> np.ndarray:
    """A filled-disk point-spread function approximating out-of-focus
    (defocus) blur - a lens's circle of confusion - rather than
    directional motion blur."""
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    total = kernel.sum()
    return kernel / total if total else kernel


def _wiener_deconvolve(gray: np.ndarray, kernel: np.ndarray, noise_ratio: float) -> np.ndarray:
    """Wiener deconvolution in the frequency domain: attempts to
    REVERSE a specific, known (here: assumed/approximate) blur kernel,
    rather than just sharpening edges the way an unsharp mask or
    kernel filter does. noise_ratio is the assumed noise-to-signal
    power ratio - the regularization term that keeps the (otherwise
    ill-posed) frequency-domain division stable; smaller values
    sharpen more aggressively but amplify noise/ringing artifacts if
    the assumed kernel doesn't match the photo's real blur, larger
    values are more conservative.

    No new dependency: implemented directly with cv2/numpy FFTs
    rather than pulling in scipy/scikit-image for this one function.
    """
    img = gray.astype(np.float32) / 255.0
    kh, kw = kernel.shape
    padded_kernel = np.zeros_like(img)
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -(kh // 2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw // 2), axis=1)

    freq_image = np.fft.fft2(img)
    freq_kernel = np.fft.fft2(padded_kernel)
    freq_kernel_conj = np.conj(freq_kernel)
    denominator = (freq_kernel * freq_kernel_conj) + noise_ratio
    freq_result = (freq_kernel_conj / denominator) * freq_image
    result = np.real(np.fft.ifft2(freq_result))
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


# One kernel per deconvolution variant - NOT a sweep across many
# angles/radii. Tried more broadly during development (multiple
# motion angles, several defocus radii) against one real held-up-photo
# sample: it did NOT recover any of the fields that were null across
# every non-deconvolution variant too (this photo's degradation looks
# more like genuine out-of-focus blur plus JPEG artifacting than a
# clean, reversible motion-blur kernel, which is exactly the case
# Wiener deconvolution is weakest at - it assumes a known, fairly
# precise kernel, and a wrong guess mostly adds ringing rather than
# removing blur). Kept anyway, at one kernel each, as cheap additional
# diversity in the ensemble for a DIFFERENT photo where the blur
# happens to be closer to one of these two shapes - each is one more
# OCR pass (bounded cost), and merge_ktp_results' per-field
# confidence rules mean a deconvolution variant that helps nothing can
# only ever be ignored, never actively chosen over a better result
# from a different variant.
_DEFOCUS_KERNEL = _defocus_blur_kernel(radius=3)
_MOTION_KERNEL = _motion_blur_kernel(length=9, angle_deg=0)


def _apply_variant_mode(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "denoise7":
        return cv2.fastNlMeansDenoising(gray, h=7)
    if mode == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(gray, -1, kernel)
    if mode == "nodenoise":
        return gray
    if mode == "deconv_defocus":
        return _wiener_deconvolve(gray, _DEFOCUS_KERNEL, noise_ratio=0.02)
    if mode == "deconv_motion":
        return _wiener_deconvolve(gray, _MOTION_KERNEL, noise_ratio=0.05)
    raise ValueError(f"unknown variant mode: {mode}")


def preprocess_for_ktp_ocr(image: np.ndarray) -> list[np.ndarray]:
    """Card-detect/crop once, then returns several differently-processed
    grayscale variants of that same crop (see module-level comment above
    for why several, not one). Falls back to a single variant - the same
    thing preprocess_for_ocr(detect_card=True) would produce - if card
    detection itself didn't find anything to crop, since in that case
    (already a flat scan) there's no held-up-photo degradation to work
    around and running 5x the OCR passes for no benefit isn't worth the
    extra latency.
    """
    cropped = _detect_and_crop_card(image)
    if cropped.shape == image.shape:
        # No crop happened - flat scan/already-cropped input. The
        # standard single-variant pipeline already handles this well.
        return [preprocess_for_ocr(image, detect_card=False)]

    # Scale is applied directly to the raw crop (matching what was
    # actually tested) - NOT on top of _upscale_if_small's own scaling,
    # which would compound into an untested, excessive final size.
    variants = []
    for scale, mode in _KTP_VARIANT_SPECS:
        resized = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = _deskew(gray)
        variants.append(_apply_variant_mode(gray, mode))
    return variants
