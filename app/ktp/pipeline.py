"""
End-to-end KTP extraction pipeline: runs the full multi-variant OCR,
whole-text field parsing, NIK region cross-check, and image-quality
assessment for one upload's pages, and returns a single merged result.

Pulled out of main.py so it can be exercised directly in tests without
spinning up FastAPI (see tests/test_extraction.py), and so main.py itself
stays a thin HTTP-layer wrapper - consistent with how ktp/parser.py /
document/parser.py / faktur_pajak/parser.py / document/signature_locator.py
already hold this service's actual logic rather than the endpoint
functions themselves.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from app.core import ocr_engine
from app.core.preprocessing import detect_and_crop_card, preprocess_for_ktp_ocr
from app.ktp import field_ocr
from app.ktp.image_quality import assess_quality
from app.ktp.parser import merge_ktp_results, parse_ktp, reconcile_nik
from app.schemas import KTPExtractionResult, QualityAssessment


class KtpPipelineResult(NamedTuple):
    result: KTPExtractionResult
    quality: QualityAssessment | None
    raw_text: str
    warnings: list[str]


def run_ktp_pipeline(pages: list[np.ndarray]) -> KtpPipelineResult:
    per_variant_results: list[KTPExtractionResult] = []
    variant_texts: list[str] = []
    nik_region_candidates: list[str] = []
    quality: QualityAssessment | None = None

    for page_index, page in enumerate(pages):
        cropped = detect_and_crop_card(page)
        card_detected = cropped.shape != page.shape
        variants = preprocess_for_ktp_ocr(page)

        if page_index == 0:
            # A KTP upload is realistically always a single page/image -
            # quality is reported once, for the page actually used to
            # judge photo quality, rather than averaged across pages a
            # real KTP submission shouldn't have.
            quality = assess_quality(page, cropped if card_detected else None, variants[0])

        for variant in variants:
            text = ocr_engine.image_to_text(variant)
            variant_texts.append(text)
            per_variant_results.append(parse_ktp(text))

            # Second, independent digits-only read of the NIK value
            # region specifically (see field_ocr.py in this same package / reconcile_nik) -
            # cheap relative to the image_to_text call above, since it's
            # a small cropped region rather than the whole page. Tried
            # at both psm 7 ("single line") and psm 8 ("single word") -
            # a NIK value crop is always both of those at once (one
            # line, one unbroken run of digits), and in practice the two
            # don't always agree with each other on a marginal crop
            # (verified against a real photo: psm 7 misread the crop's
            # last digit while psm 8, on the exact same crop, read it
            # correctly) - rather than guess which mode is more reliable
            # in general, both get a vote and reconcile_nik's own
            # agreement-vs-disagreement handling sorts it out, the same
            # way it already does for the preprocessing-variant ensemble.
            words = ocr_engine.image_to_data(variant)
            value_crop = field_ocr.locate_nik_value_crop(variant, words)
            if value_crop is not None:
                for psm in (7, 8):
                    digits, _conf = ocr_engine.image_to_digits(value_crop, psm=psm)
                    if digits:
                        nik_region_candidates.append(digits)

    merged = merge_ktp_results(per_variant_results)
    merged = merged.model_copy(update={"nik": reconcile_nik(merged.nik, nik_region_candidates)})

    raw_text = "\n----- next preprocessing variant -----\n".join(variant_texts)
    warnings = _quality_warnings(quality)
    return KtpPipelineResult(result=merged, quality=quality, raw_text=raw_text, warnings=warnings)


def _quality_warnings(quality: QualityAssessment | None) -> list[str]:
    if quality is None:
        return []
    warnings: list[str] = []
    if not quality.document_detected:
        warnings.append(
            "Could not automatically detect a distinct KTP card boundary in the photo - processed "
            "the full image as-is. Expected and harmless if this upload is already a flat scan/crop "
            "of just the card; if it's a photo of someone holding the card up, a flatter angle "
            "against a plain background usually helps."
        )
    if quality.blur_score < 0.3:
        warnings.append("KTP detected but the source image appears blurry.")
    if not quality.resolution_ok:
        warnings.append("KTP resolution is low relative to the recommended minimum - consider asking for a higher-resolution photo.")
    if quality.glare_detected:
        warnings.append("Strong glare/overexposure detected on the card.")
    elif not quality.lighting_ok:
        warnings.append("Uneven or poor lighting detected on the card.")
    return warnings
