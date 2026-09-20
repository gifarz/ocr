"""
CompliFi OCR Service - a standalone FastAPI microservice.

This is deliberately NOT part of the CompliFi Node/Express backend or the
Next.js frontend. It has no knowledge of CompliFi's database, sessions,
Kong gateway, or tenants - it's a pure function-as-a-service: send a file,
get structured fields back. CompliFi's backend calls this over HTTP (see
README.md for the integration snippet) and decides what to do with the
result; this service never talks to Peruri, never writes to CompliFi's
database, and never stores the uploaded file beyond the request.

Run locally:
    uvicorn app.main:app --reload --port 8088

Run in production (e.g. alongside the existing PM2-managed CompliFi
processes):
    pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8088" --name complifi-ocr
"""

from __future__ import annotations

import time
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import fitz  # PyMuPDF

from app.config import settings
from app.core import ocr_engine
from app.core.pdf_text import extract_full_text, has_extractable_text
from app.core.preprocessing import (
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_PDF_TYPE,
    load_pages,
    preprocess_for_ocr,
)
from app.document.parser import parse_document
from app.document.signature_locator import locate_signature_box
from app.faktur_pajak.parser import parse_faktur_pajak
from app.ktp.pipeline import run_ktp_pipeline
from app.schemas import ExtractionResponse, HealthResponse, KTPExtractionResult, SignatureLocateResponse

app = FastAPI(
    title="CompliFi OCR Service",
    description="Standalone OCR microservice for KTP identity extraction and document field extraction.",
    version="0.1.0",
)

# CORS is permissive here because this service is meant to be called
# server-to-server (CompliFi's Express backend), not directly from a
# browser. If you do expose it to a browser, tighten this to the actual
# CompliFi origin(s).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"])


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        return  # auth disabled - local dev only, see config.py
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


async def _read_and_validate(file: UploadFile) -> bytes:
    if file.content_type not in SUPPORTED_IMAGE_TYPES and file.content_type != SUPPORTED_PDF_TYPE:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Send a PDF or an image.",
        )
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return data


def _ocr_all_pages(file_bytes: bytes, content_type: str, detect_card: bool = False) -> tuple[str, int]:
    pages = load_pages(file_bytes, content_type)
    texts = []
    for page in pages:
        processed = preprocess_for_ocr(page, detect_card=detect_card)
        texts.append(ocr_engine.image_to_text(processed))
    return "\n".join(texts), len(pages)


def _extract_text_preferring_native(file_bytes: bytes, content_type: str) -> tuple[str, int, bool]:
    """Returns (raw_text, pages_processed, used_native_pdf_text).

    A Faktur Pajak is almost always a digitally generated PDF (Coretax,
    or any e-Faktur issuer) - reading its own embedded text via PyMuPDF
    is both faster and far more accurate than rasterizing it and
    running Tesseract, the same reasoning /v1/locate-signature already
    applies (see core/pdf_text.py). This only falls back to the OCR pipeline
    for a genuinely scanned/photographed invoice, or for a non-PDF
    image upload.
    """
    if content_type == SUPPORTED_PDF_TYPE:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            if has_extractable_text(doc):
                return extract_full_text(doc), doc.page_count, True
    raw_text, pages = _ocr_all_pages(file_bytes, content_type)
    return raw_text, pages, False


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        tesseract_version=ocr_engine.tesseract_version(),
        languages=ocr_engine.available_languages(),
    )


@app.post("/v1/extract/ktp", response_model=ExtractionResponse, dependencies=[Depends(require_api_key)])
async def extract_ktp(file: UploadFile = File(...), include_raw_text: bool = False) -> ExtractionResponse:
    """Extracts NIK/name/etc. from a photo (or scan) of an Indonesian KTP.

    Intended to pre-fill the `identityNumber` / `clientName` fields on
    CompliFi's stamp form - NOT to be trusted as a final, unverified
    source of truth. The caller should always show these as editable,
    reviewable values (see FieldResult.confidence).

    Runs OCR against SEVERAL differently-preprocessed versions of the
    same photo (preprocess_for_ktp_ocr - denoise strength, sharpening,
    and upscale factor each varied) and merges the results, taking the
    best candidate independently per field (merge_ktp_results) - a real
    degraded source photo (someone holding the card up, rather than a
    flat scan) showed no single preprocessing choice recovering more
    than about half the card, with little overlap between which half
    each one got right. This costs several times the OCR latency of a
    single pass; preprocess_for_ktp_ocr already skips the ensemble
    (falls back to one pass) when card detection finds nothing to crop,
    since a flat scan doesn't have this problem to begin with.

    NIK additionally gets a second, independent digits-only OCR pass
    against just the cropped value region next to the "NIK" label
    (ktp/field_ocr.py), cross-checked against the whole-text tiered read
    (ktp_parser.reconcile_nik) - see ktp_pipeline.run_ktp_pipeline, which
    holds this endpoint's actual logic.

    Also runs a lightweight image-quality assessment (blur, resolution,
    lighting/glare, whether a card was even detected) on the source photo
    - surfaced as `quality` in the response and folded into `warnings`,
    so a thin/null-heavy result can be told apart from a parser bug.
    """
    start = time.monotonic()
    data = await _read_and_validate(file)
    pages = load_pages(data, file.content_type)

    pipeline = run_ktp_pipeline(pages)
    result = pipeline.result
    raw_text = pipeline.raw_text

    warnings = list(pipeline.warnings)
    if result.nik.source == "not_found":
        warnings.append("Could not locate a 16-digit NIK. Ask the user to re-scan with better lighting/focus.")
    elif result.nik.confidence < 0.6:
        warnings.append("NIK was recovered but wasn't confirmed by a second, independent read - have the user double check it.")

    low_confidence_fields = [
        name for name in KTPExtractionResult.model_fields
        if (fr := getattr(result, name)).value is not None and fr.confidence < 0.5
    ]
    if low_confidence_fields:
        warnings.append(f"Low-confidence fields worth a closer look: {', '.join(low_confidence_fields)}.")

    missing_fields = [name for name in KTPExtractionResult.model_fields if getattr(result, name).value is None]
    if missing_fields:
        warnings.append(
            f"Not recognized even after multiple preprocessing attempts: {', '.join(missing_fields)}. "
            "The source photo may be too blurry, low-resolution, or at too sharp an angle for these "
            "specific fields - ask the user for a clearer photo if these are required."
        )

    return ExtractionResponse(
        request_id=str(uuid.uuid4()),
        processing_ms=int((time.monotonic() - start) * 1000),
        pages_processed=len(pages),
        raw_text_included=include_raw_text,
        raw_text=raw_text if include_raw_text else None,
        ktp=result,
        quality=pipeline.quality,
        warnings=warnings,
    )


@app.post("/v1/extract/document", response_model=ExtractionResponse, dependencies=[Depends(require_api_key)])
async def extract_document(file: UploadFile = File(...), include_raw_text: bool = False) -> ExtractionResponse:
    """Extracts document date/value candidates from a contract/invoice PDF
    to pre-fill `docdate` / `docvalue` on CompliFi's stamp form."""
    start = time.monotonic()
    data = await _read_and_validate(file)
    raw_text, pages = _ocr_all_pages(data, file.content_type)
    result = parse_document(raw_text)

    warnings = []
    if not result.docdate_candidates:
        warnings.append("No dates recognized in the document text.")
    if not result.docvalue_candidates:
        warnings.append("No currency amounts recognized in the document text.")
    if len(result.docdate_candidates) > 1:
        warnings.append(f"{len(result.docdate_candidates)} possible dates found - confirm the right one with the user.")
    if len(result.docvalue_candidates) > 1:
        warnings.append(f"{len(result.docvalue_candidates)} possible amounts found - confirm the right one with the user.")

    return ExtractionResponse(
        request_id=str(uuid.uuid4()),
        processing_ms=int((time.monotonic() - start) * 1000),
        pages_processed=pages,
        raw_text_included=include_raw_text,
        raw_text=raw_text if include_raw_text else None,
        document=result,
        warnings=warnings,
    )


@app.post("/v1/extract/faktur-pajak", response_model=ExtractionResponse, dependencies=[Depends(require_api_key)])
async def extract_faktur_pajak(file: UploadFile = File(...), include_raw_text: bool = False) -> ExtractionResponse:
    """Extracts header (FK) and line-item (OF) fields from an Indonesian
    Tax Invoice (Faktur Pajak), field-named to match CompliFi's
    bulk-import CSV columns.

    Prefers the PDF's own embedded text over OCR whenever the upload is
    a digitally generated PDF (the normal case for a Coretax/e-Faktur
    invoice) - see _extract_text_preferring_native. Falls back to the
    Tesseract pipeline only for a scanned/photographed invoice or a
    plain image upload, in which case `used_native_pdf_text` is False
    and results should be treated as lower-confidence, same as every
    other field this service returns (never auto-submit without review;
    see FieldResult / FakturPajakExtractionResult.validation).

    Per-line tax fields (DPP/DPP_LAIN/TARIF_PPN/PPN/TARIF_PPNBM/PPNBM/
    CHECK_DPP_LAIN) are only populated when the document itself prints
    a per-line breakdown - never allocated pro-rata from the invoice
    totals. See faktur_pajak/parser.py.
    """
    start = time.monotonic()
    data = await _read_and_validate(file)
    raw_text, pages, used_native_pdf_text = _extract_text_preferring_native(data, file.content_type)
    result = parse_faktur_pajak(raw_text)

    warnings = []
    if not used_native_pdf_text:
        warnings.append(
            "No extractable text found in the PDF (or a plain image was uploaded) - fell back "
            "to OCR. Treat every field as lower-confidence and verify against the original scan."
        )
    if not result.nomor_faktur:
        warnings.append("Could not locate a Faktur Pajak number (expected format NNN.NNN-NN.NNNNNNNN).")
    if not result.line_items:
        warnings.append("No line items recognized - check the invoice's item table formatting.")
    if result.validation.calculation_status == "WARNING":
        warnings.append("Calculation validation found discrepancies - see validation.calculation_notes.")
    if any(item.dpp is None for item in result.line_items):
        warnings.append(
            "This invoice only prints tax totals at the invoice level - per-line DPP/PPN fields "
            "are left null rather than split pro-rata (no confirmed allocation rule yet)."
        )

    return ExtractionResponse(
        request_id=str(uuid.uuid4()),
        processing_ms=int((time.monotonic() - start) * 1000),
        pages_processed=pages,
        raw_text_included=include_raw_text,
        raw_text=raw_text if include_raw_text else None,
        faktur_pajak=result,
        warnings=warnings,
    )


@app.post("/v1/locate-signature", response_model=SignatureLocateResponse, dependencies=[Depends(require_api_key)])
async def locate_signature(
    file: UploadFile = File(...),
    signer_name: str = Form(...),
) -> SignatureLocateResponse:
    """Finds where to place `signer_name`'s signature specimen: the
    blank cell directly above their printed name in a signature block
    (e.g. the "Pemohon" column on CompliFi's Peruri-style forms).

    `signer_name` should be the signer's full legal name as CompliFi
    has it (e.g. from the KTP extraction or the tenant's user record) -
    matching tolerates the document abbreviating it ("Muhammad Gifar
    Z." still matches "Muhammad Gifar Zaini").

    On a digitally generated PDF (the normal case for CompliFi/Peruri
    documents) this reads the file's own text and table-rule objects
    directly and returns exact PDF-point coordinates - no OCR involved.
    It only falls back to OCR + pixel-based line detection for a
    genuinely scanned/photographed document, in which case `method` on
    each match will say "ocr_scanned" and confidence will be lower;
    treat those results as pre-fill for a human to confirm, the same as
    every other field this service returns (see FieldResult).

    A document can have the same signer's name on more than one page
    (e.g. a multi-page contract re-signed per page, or - as seen in a
    sample upload - the same page duplicated across a PDF); this
    returns a match per page rather than assuming there's exactly one.
    """
    start = time.monotonic()
    data = await _read_and_validate(file)
    matches = locate_signature_box(data, file.content_type, signer_name)

    warnings = []
    if not matches:
        warnings.append(
            f"Could not find a signature block matching '{signer_name}'. The name may be "
            "spelled/abbreviated differently on the document, or this document doesn't use "
            "a signature table this service recognizes - have the user place it manually."
        )

    return SignatureLocateResponse(
        request_id=str(uuid.uuid4()),
        processing_ms=int((time.monotonic() - start) * 1000),
        signer_name=signer_name,
        matches=matches,
        warnings=warnings,
    )
