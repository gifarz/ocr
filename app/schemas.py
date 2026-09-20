from typing import Literal, Optional

from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    """One extracted field plus how sure we are about it.

    `confidence` is a simple heuristic score (0-1), not a statistical
    probability - see extraction.py for how each field computes it.
    CompliFi's frontend should treat anything below ~0.6 as "needs review"
    and never auto-submit a field the user hasn't seen.
    """

    value: Optional[str] = None
    confidence: float = 0.0
    source: Literal["ocr", "not_found"] = "not_found"


class QualityAssessment(BaseModel):
    """Diagnostic signals about the SOURCE PHOTO, computed before/alongside
    OCR - not a confidence measure itself (see FieldResult.confidence for
    that). Lets a caller (or a human reviewing a low-confidence result)
    tell "the parser missed something readable" apart from "the photo
    genuinely didn't have enough information", backed by concrete numbers
    instead of a wall of nulls. Only populated on `/v1/extract/ktp` today;
    None on every other endpoint's response.
    """

    document_detected: bool
    perspective_corrected: bool
    blur_score: float = Field(ge=0.0, le=1.0)
    resolution_ok: bool
    lighting_ok: bool
    glare_detected: bool
    document_area_ratio: Optional[float] = None


class KTPExtractionResult(BaseModel):
    nik: FieldResult = FieldResult()
    nama: FieldResult = FieldResult()
    tempat_lahir: FieldResult = FieldResult()
    tanggal_lahir: FieldResult = FieldResult()
    jenis_kelamin: FieldResult = FieldResult()
    golongan_darah: FieldResult = FieldResult()
    alamat: FieldResult = FieldResult()
    rt_rw: FieldResult = FieldResult()
    kel_desa: FieldResult = FieldResult()
    kecamatan: FieldResult = FieldResult()
    agama: FieldResult = FieldResult()
    status_perkawinan: FieldResult = FieldResult()
    pekerjaan: FieldResult = FieldResult()
    kewarganegaraan: FieldResult = FieldResult()
    berlaku_hingga: FieldResult = FieldResult()


class DocumentExtractionResult(BaseModel):
    """Fields relevant to the e-Meterai stamp form (docdate / docvalue),
    extracted from a generic document rather than a KTP."""

    docdate: FieldResult = FieldResult()
    docvalue: FieldResult = FieldResult()
    # All date-like and currency-like strings found, in case the top pick
    # is wrong and the frontend wants to offer alternatives in a dropdown.
    docdate_candidates: list[str] = Field(default_factory=list)
    docvalue_candidates: list[str] = Field(default_factory=list)


class FakturPajakLineItem(BaseModel):
    """One `OF` (object/line-item) record of a Faktur Pajak.

    Per-line tax fields (check_dpp_lain/dpp/dpp_lain/tarif_ppn/ppn/
    tarif_ppnbm/ppnbm) are only populated when the document itself
    prints a per-line breakdown - the common case is invoice-level
    totals only, in which case these stay None rather than being
    allocated pro-rata from the FK totals (no confirmed business rule
    for that yet - see faktur_pajak/parser.py)."""

    barang_jasa: Optional[str] = None
    kode_objek: Optional[str] = None
    nama: Optional[str] = None
    satuan: Optional[str] = None
    harga_satuan: Optional[float] = None
    jumlah_barang: Optional[float] = None
    harga_total: Optional[float] = None
    diskon: Optional[float] = None
    check_dpp_lain: Optional[bool] = None
    dpp: Optional[float] = None
    dpp_lain: Optional[float] = None
    tarif_ppn: Optional[float] = None
    ppn: Optional[float] = None
    tarif_ppnbm: Optional[float] = None
    ppnbm: Optional[float] = None


class FakturPajakValidation(BaseModel):
    calculation_status: Literal["VALID", "WARNING", "INVALID", "NOT_CHECKED"] = "NOT_CHECKED"
    calculation_notes: list[str] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class FakturPajakExtractionResult(BaseModel):
    """One `FK` (header) record plus its `OF` (line-item) records,
    field-named to match CompliFi's bulk-import CSV columns
    (lower_snake_case here; the CSV template uses UPPER_SNAKE_CASE -
    see README's CSV column mapping table).

    Every field is None when not found in the document - never
    invented, assumed, or derived - matching the extraction spec this
    was built from."""

    npwp_wp: Optional[str] = None
    id_tku_wp: Optional[str] = None
    kd_jenis_transaksi: Optional[str] = None
    fg_pengganti: Optional[str] = None
    nomor_faktur: Optional[str] = None
    masa_pajak: Optional[str] = None
    tahun_pajak: Optional[str] = None
    tanggal_faktur: Optional[str] = None
    npwp: Optional[str] = None
    jenis_identitas: Optional[str] = None
    nik_nomor_passport: Optional[str] = None
    kode_negara: Optional[str] = None
    nama: Optional[str] = None
    email_pembeli: Optional[str] = None
    alamat_pembeli: Optional[str] = None
    tku_pembeli: Optional[str] = None
    jumlah_dpp: Optional[float] = None
    jumlah_dpp_lain: Optional[float] = None
    jumlah_ppn: Optional[float] = None
    jumlah_ppnbm: Optional[float] = None
    referensi: Optional[str] = None
    fg_uang_muka: Optional[str] = None
    nomor_faktur_um_sebelumnya: Optional[str] = None
    uang_muka_dpp: Optional[float] = None
    uang_muka_dpp_lain: Optional[float] = None
    uang_muka_ppn: Optional[float] = None
    uang_muka_ppnbm: Optional[float] = None
    kode_dokumen_pendukung: Optional[str] = None
    tax_code: Optional[str] = None
    faktur_pajak_dummy: Optional[bool] = None
    line_items: list[FakturPajakLineItem] = Field(default_factory=list)
    validation: FakturPajakValidation = Field(default_factory=FakturPajakValidation)


class SignatureBoxResult(BaseModel):
    """Where a signer's signature specimen should be placed: the blank
    cell directly above their printed name in a signature block (e.g.
    the "Pemohon" column on CompliFi's Peruri-style forms).

    Coordinates are in the document's own space, NOT a fixed pixel
    grid: PDF points with a top-left origin when `method` is
    "pdf_native" (the common case for CompliFi/Peruri-generated
    documents), or pixel coordinates at whatever DPI the page was
    rasterized at when `method` is "ocr_scanned". Always check `method`
    and use `page_width`/`page_height` to convert before stamping.
    """

    page_index: int
    matched_text: str
    match_confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    method: Literal["pdf_native", "ocr_scanned"]


class SignatureLocateResponse(BaseModel):
    request_id: str
    processing_ms: int
    signer_name: str
    matches: list[SignatureBoxResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    request_id: str
    processing_ms: int
    pages_processed: int
    raw_text_included: bool = False
    raw_text: Optional[str] = None
    ktp: Optional[KTPExtractionResult] = None
    document: Optional[DocumentExtractionResult] = None
    faktur_pajak: Optional[FakturPajakExtractionResult] = None
    quality: Optional[QualityAssessment] = None
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    tesseract_version: str
    languages: list[str]
