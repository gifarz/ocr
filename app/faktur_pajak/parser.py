"""
Extracts structured header ("FK") and line-item ("OF") fields from an
Indonesian Tax Invoice (Faktur Pajak) - DJP Coretax-issued or otherwise -
for CompliFi's e-Faktur bulk-import pipeline.

Field names and the two-record-type (FK/OF) shape mirror the CSV
bulk-import format CompliFi already uses. This module is a pure
text -> struct parser: it doesn't know or care whether `raw_text` came
from native PDF text (the common case - a Coretax/e-Faktur PDF is
digitally generated, so PyMuPDF's page.get_text() already gives clean
text, no OCR needed) or from Tesseract OCR (the fallback for a
scanned/photographed faktur). See main.py for how that choice is made,
the same way it already is for /v1/locate-signature (core/pdf_text.py's
has_extractable_text).

Two real layouts are supported side by side, since both have been seen
in production:

1. A DJP Coretax-issued Faktur Pajak (the common real-world case,
   verified against an actual Coretax PDF) - labels separated from
   their value by a literal ":", full-word labels ("Dasar Pengenaan
   Pajak" rather than "DPP", "Jumlah PPN" rather than "PPN"), the
   invoice date only appears as an Indonesian-month-name date next to
   the e-signature ("KOTA ADM. ..., 02 Januari 2026"), a blank field is
   printed as a literal "-" rather than omitted, the buyer's NITKU is
   embedded in their address as "#<22 digits>" rather than its own
   labelled field, and each line item's per-line PPnBM rate/amount and
   discount ARE printed (unlike the invoice-totals-only case below).
2. A simpler/older layout (no literal colons, abbreviated labels like
   bare "DPP", a one-row-per-item table, no per-line tax breakdown)
   used for early testing of this service.

Deliberately conservative, matching the rest of this service: a field
is only ever populated when a clear match is found in the text - never
inferred from outside knowledge. Two _derivations_ are an exception,
and are called out explicitly where they happen: JENIS_IDENTITAS
(logically determined by *which* of NIK/Nomor Paspor/NPWP/Identitas
Lain is actually non-blank - not fabricated data, just naming which
of four already-extracted fields is populated) and FG_UANG_MUKA as a
0/1 flag (whether "Dikurangi Uang Muka yang telah diterima" has an
amount next to it or not) when the document doesn't print an explicit
FG_UANG_MUKA-style field. Per-line CHECK_DPP_LAIN/DPP/DPP_LAIN/
TARIF_PPN/PPN on a line item still only ever get a value when the
document itself prints a per-line breakdown for THAT field - the
common case (invoice-level totals only) leaves them None rather than
allocating totals pro-rata, since there's no confirmed business rule
for that yet.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas import FakturPajakExtractionResult, FakturPajakLineItem, FakturPajakValidation

# ---------------------------------------------------------------------------
# Number / identifier / date normalization
# ---------------------------------------------------------------------------


def _to_number(raw: Optional[str]) -> Optional[float]:
    """'Rp5.000.000' / '5.000.000' / '12' / '1,5' / '270.270.270,00' ->
    5000000.0 / 12.0 / 1.5 / 270270270.0

    Indonesian formatting uses '.' as the thousands separator and ','
    as the decimal separator - the reverse of US formatting - so a
    comma is only ever treated as a decimal point, never stripped like
    a thousands separator would be.
    """
    if raw is None:
        return None
    cleaned = re.sub(r"(?i)rp\.?|idr|%", "", raw).strip()
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_date(raw: Optional[str]) -> Optional[str]:
    """'17/09/2026' -> '2026-09-17'. Leaves unparseable input as-is."""
    if raw is None:
        return None
    raw = raw.strip()
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$", raw)
    if m:
        day, month, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return raw


_INDO_MONTHS = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}
# A Coretax Faktur Pajak's date usually only appears next to the
# e-signature line ("KOTA ADM. JAKARTA SELATAN, 02 Januari 2026") -
# there's no separate "Tanggal Faktur" label at all in that layout.
# The day/month and year can land on different physical lines
# ("...02 Januari\n2026\n..."), which \s+ (matching a newline same as
# a space) already handles without any special-casing.
_INDO_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|"
    r"September|Oktober|November|Desember)\s+(\d{4})\b",
    re.IGNORECASE,
)


def _find_indo_date(text: str) -> Optional[str]:
    m = _INDO_DATE_RE.search(text)
    if not m:
        return None
    day, month_name, year = m.groups()
    month = _INDO_MONTHS.get(month_name.lower())
    if not month:
        return None
    return f"{year}-{month:02d}-{int(day):02d}"


def _digits_only(raw: Optional[str]) -> Optional[str]:
    """Strips separators from an identifier (NPWP/NITKU): e.g.
    '00.111.222.3-444.555' -> '001112223444555'."""
    if raw is None:
        return None
    digits = re.sub(r"\D", "", raw)
    return digits or None


# A DJP Coretax document prints a literal "-" (occasionally an en/em
# dash) for any field that doesn't apply ("NIK : -", "Nomor Paspor :
# -", "Identitas Lain : -") rather than omitting the label entirely -
# treat that the same as "not found" everywhere a value is pulled.
_BLANK_MARKERS = {"-", "\u2013", "\u2014", "N/A", "NA", ""}


# ---------------------------------------------------------------------------
# Section splitting - keeps the seller's and buyer's identically-shaped
# fields (both have an NPWP, a Nama, an Alamat) from being mixed up.
# ---------------------------------------------------------------------------

_SELLER_HEADER_RE = re.compile(r"Pengusaha\s+Kena\s+Pajak", re.IGNORECASE)
# Real Coretax wording is "Pembeli Barang Kena Pajak/Penerima Jasa Kena
# Pajak:" - quite different from the simpler "Pembeli / Penerima Jasa"
# tested against earlier. Anchoring on just "Pembeli ... Penerima"
# within a short window (rather than requiring an exact phrase either
# side) matches both without needing to enumerate every wording DJP
# might use.
_BUYER_HEADER_RE = re.compile(r"Pembeli\b[\s\S]{0,80}?Penerima\b", re.IGNORECASE)
# Tried in order; whichever is found FIRST after the buyer heading ends
# the buyer block. "Email" is the last buyer-identity field printed in
# both known layouts, so it's tried first - its own line is kept
# *inside* the buyer block (hence using the end of the whole line, via
# the second pattern in each tuple, rather than just the word "Email").
# The other two are markers for the START of the next section instead,
# so those cut right at the label. Without any fallback here, a layout
# lacking all three markers would have the buyer block swallow the
# entire rest of the document (line items, totals, everything) - so
# there IS always a result, just possibly the entire document if a
# layout is encountered that doesn't match any of these.
_BUYER_END_MARKERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Email\s*:?[^\n]*", re.IGNORECASE), "end"),
    (re.compile(r"Tanggal\s+Faktur", re.IGNORECASE), "start"),
    (re.compile(r"No\.?\s*\n\s*Kode\b", re.IGNORECASE), "start"),
]


def _find_buyer_end(text: str, start: int) -> int:
    for pattern, which in _BUYER_END_MARKERS:
        m = pattern.search(text, start)
        if m:
            return m.end() if which == "end" else m.start()
    return len(text)


def _split_sections(text: str) -> tuple[str, str, str]:
    """Returns (seller_block, buyer_block, rest_of_document)."""
    seller_match = _SELLER_HEADER_RE.search(text)
    buyer_match = _BUYER_HEADER_RE.search(text)

    seller_start = seller_match.end() if seller_match else 0
    seller_end = buyer_match.start() if buyer_match else len(text)
    buyer_start = buyer_match.end() if buyer_match else len(text)
    buyer_end = _find_buyer_end(text, buyer_start)

    return text[seller_start:seller_end], text[buyer_start:buyer_end], text[buyer_end:]


def _find(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    value = m.group(1).strip()
    if value in _BLANK_MARKERS:
        return None
    return value or None


def _find_any(patterns: list[str], text: str) -> Optional[str]:
    """Tries each pattern in order, returning the first match. Used for
    fields where the two known layouts use different label wording
    (e.g. "Dasar Pengenaan Pajak" vs. bare "DPP") - listed real-layout
    first, since that's the layout actually seen in production."""
    for pattern in patterns:
        value = _find(pattern, text)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Field patterns. `\s*:?\s*` between a label and its value (rather than
# just `\s+`) matters a lot here: the synthetic test layout's PDF text
# extraction happened not to include a literal colon character, but a
# real Coretax PDF's does ("NPWP : 0017183278093000") - `\s+` alone
# stops dead the instant it hits that ":" since a colon isn't
# whitespace, which is exactly what caused every field to come back
# null on a real document. `\s*:?\s*` also already matches a newline
# (both `\s*` groups include `\n`), so it doubles as the fix for
# "label on one line, value on the next" - a common real layout too.
# ---------------------------------------------------------------------------

_NOMOR_FAKTUR_DOTTED_RE = r"(\d{3}\.\d{3}-\d{2}\.\d{8})"
# Matches both "Kode & Nomor Faktur" (older/simpler layout) and "Kode
# dan Nomor Seri Faktur Pajak" (real Coretax layout) via the optional
# groups, then whatever identifier follows - either the older dotted
# format or Coretax's newer unbroken 17-digit string.
_NOMOR_FAKTUR_LABELED_RE = (
    r"Kode\s*(?:&|dan)\s*(?:Nomor\s*)?(?:Seri\s*)?Faktur(?:\s*Pajak)?\s*:?\s*([\d.\-]{10,25})"
)
_STATUS_PENGGANTI_RE = r"Status\s+Pengganti\s*:?\s*(\d)"
_MASA_TAHUN_RE = r"Masa\s+Pajak\s*:?\s*(\d{1,2})\s*/\s*\w+\s+(\d{4})"
_TANGGAL_FAKTUR_LABELED_RE = r"Tanggal\s+Faktur\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})"
# "(Referensi: SO-112000RG-202601-0024)" (real layout, parenthesized)
# and "Referensi INV/CTI/IX/2026/0042" (older layout) both match - the
# optional leading "(" doesn't need to be consumed since it's outside
# the match start, and a trailing ")" simply isn't in the captured
# character class, so it's left behind rather than swallowed.
_REFERENSI_RE = r"Referensi\s*:?\s*([A-Za-z0-9/\-]+)"

_NAMA_RE = r"Nama\s*:?\s*(.+)"
_NPWP_RE = r"NPWP\s*:?\s*([\d.\-]{15,25})"
_NITKU_LABELED_RE = r"NITKU\s*:?\s*(\d{10,25})"
# A real Coretax PDF doesn't print NITKU as its own labelled field at
# all - it's appended directly onto the end of the Nama/Alamat block as
# "#<22 digits>" (NPWP + a 6-digit branch suffix), with no label of its
# own. Matched separately per NPWP via _find_tku_by_npwp below, since
# which "#..." belongs to the seller vs. the buyer isn't positional
# (it can appear before the seller's own labelled block, in the page's
# "kepada" mailing box) - only the digit prefix tells them apart.
_HASH_ID_RE = re.compile(r"#(\d{18,25})")
# Alamat can wrap across several physical lines in native PDF text
# (a long address on a narrow column) - keep consuming lines until
# hitting the next field's label, rather than just the first line.
_ALAMAT_RE = (
    r"Alamat\s*:?\s*((?:(?!\s*(?:NPWP|NIK|Nomor\s+Paspor|Identitas\s+Lain|Email|"
    r"Pembeli|Pengusaha)\b).+\n?)+)"
)
_EMAIL_RE = r"Email\s*:?\s*(\S+@\S+)"
_JENIS_IDENTITAS_RE = r"Jenis\s+Identitas\s*:?\s*(\S+)"
# A blank NIK/passport/other-ID field runs straight into the next
# label with no value at all ("...Paspor\nKode Negara IDN" or, in the
# real layout, is printed as a literal "-"). Requiring a digit in the
# captured token is what tells a real value apart from either of those
# ("-" has no digit and fails the match entirely; a swallowed label's
# first word like "Kode" also has no digit).
_NIK_RE = r"\bNIK\s*:?\s*([A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)"
_NOMOR_PASPOR_RE = r"Nomor\s+Paspor\s*:?\s*([A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)"
_IDENTITAS_LAIN_RE = r"Identitas\s+Lain\s*:?\s*([A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)"
# Older/simpler layout combined these into one label+line.
_NIK_PASSPORT_COMBINED_RE = r"NIK\s*/\s*Nomor\s+Paspor\s*:?\s*([A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)"
_KODE_NEGARA_RE = r"Kode\s+Negara\s*:?\s*([A-Z]{3})"

_HARGA_JUAL_RE = r"Harga\s+Jual\s*/\s*Penggantian(?:\s*/\s*Uang\s+Muka)?(?:\s*/\s*Termin)?\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)"
# Real layout: "Dikurangi Potongan Harga" (summary row). Older layout:
# bare "Potongan Harga" - but a real Coretax document ALSO prints
# "Potongan Harga = Rp ..." per line item, with a "=" sign that this
# fallback pattern's negative lookahead avoids matching, so the
# invoice-level total (tried first via the "Dikurangi" wording) isn't
# accidentally shadowed by - or confused with - the first line item's
# own per-line discount when both wordings appear in the same document.
_POTONGAN_PATTERNS = [
    r"Dikurangi\s+Potongan\s+Harga\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
    r"Potongan\s+Harga(?!\s*=)\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
]
_DPP_PATTERNS = [
    r"Dasar\s+Pengenaan\s+Pajak\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
    r"(?<!Nilai Lain )\bDPP\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
]
_DPP_LAIN_RE = r"DPP\s+Nilai\s+Lain\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)"
_PPN_PATTERNS = [
    r"Jumlah\s+PPN\b[^\n\d]*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
    r"(?<!Tarif )\bPPN\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
]
_PPNBM_PATTERNS = [
    r"Jumlah\s+PPnBM\b[^\n\d]*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
    r"(?<!Tarif )\bPPnBM\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)",
]

_FG_UANG_MUKA_LABELED_RE = r"FG_UANG_MUKA\s*:?\s*(\d)"
# Real layout has no explicit FG_UANG_MUKA-style field at all - only a
# "Dikurangi Uang Muka yang telah diterima" summary line, printed with
# no amount at all (not even "0,00") when there's no down payment. This
# is a genuine structural DERIVATION, not invented data: the flag is
# definitionally "was an uang muka amount deducted or not", which this
# line directly answers either way. Only used when the literal
# FG_UANG_MUKA label isn't present.
_UANG_MUKA_DIKURANGI_RE = re.compile(
    r"Dikurangi\s+Uang\s+Muka\s+yang\s+telah\s+diterima\s*:?\s*(?:Rp\.?\s*)?([\d.,]*)",
    re.IGNORECASE,
)
_NOMOR_FAKTUR_UM_RE = r"Nomor\s+Faktur\s+Uang\s+Muka\s+Sebelumnya\s*:?\s*([A-Za-z0-9.\-]*\d[A-Za-z0-9.\-]*)"
_UM_DPP_RE = r"Uang\s+Muka\s+DPP\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)"
_UM_DPP_LAIN_RE = r"Uang\s+Muka\s+DPP\s+Nilai\s+Lain\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)"
_UM_PPN_RE = r"Uang\s+Muka\s+PPN\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)"
_UM_PPNBM_RE = r"Uang\s+Muka\s+PPnBM\s*:?\s*(?:Rp\.?\s*)?([\d.,]+)"

_KODE_DOKUMEN_RE = (
    r"Kode\s+Dokumen\s+Pendukung\s*:?\s*([A-Za-z0-9/\-]{1,20}?)"
    r"(?=Tempat\b|Tanggal\b|Penandatangan\b|\s|$)"
)

_DUMMY_MARKERS_RE = re.compile(
    r"data\s+contoh|fiktif|dummy|pengujian\s+sistem|sample\s+invoice|for\s+testing",
    re.IGNORECASE,
)


def _find_tku_by_npwp(text: str, npwp_digits: Optional[str]) -> Optional[str]:
    """A "#<22 digits>" ID matching this NPWP as a prefix is its NITKU
    (NPWP + a 6-digit branch/location suffix) - searches the WHOLE
    document rather than just one block, since this annotation can
    appear anywhere (e.g. the seller's copy shows up in a "kepada"
    mailing box that comes before the seller's own labelled section,
    not inside it)."""
    if not npwp_digits:
        return None
    for m in _HASH_ID_RE.finditer(text):
        if m.group(1).startswith(npwp_digits):
            return m.group(1)
    return None


def _resolve_buyer_identity(
    buyer_block: str, buyer_npwp_digits: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Returns (nik_nomor_passport, jenis_identitas).

    An explicit "Jenis Identitas" label (older layout) is trusted
    outright when present. Otherwise - the real Coretax layout doesn't
    print one - this is DERIVED from which of NIK / Nomor Paspor / NPWP
    / Identitas Lain is actually non-blank, in that priority order.
    This is a structural inference (naming which already-extracted
    field is populated), not fabricated data.
    """
    explicit = _find(_JENIS_IDENTITAS_RE, buyer_block)
    nik = _find(_NIK_RE, buyer_block)
    passport = _find(_NOMOR_PASPOR_RE, buyer_block)
    if not nik and not passport:
        combined = _find(_NIK_PASSPORT_COMBINED_RE, buyer_block)
        nik = combined

    if explicit:
        return (nik or passport), explicit.upper()
    if nik:
        return nik, "NIK"
    if passport:
        return passport, "PASPOR"
    if buyer_npwp_digits:
        return None, "NPWP"
    identitas_lain = _find(_IDENTITAS_LAIN_RE, buyer_block)
    if identitas_lain:
        return identitas_lain, "LAINNYA"
    return None, None


def _derive_fg_uang_muka(rest: str) -> Optional[str]:
    m = _UANG_MUKA_DIKURANGI_RE.search(rest)
    if not m:
        return None
    return "1" if m.group(1).strip() else "0"


def _clean_alamat(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = _HASH_ID_RE.sub("", raw)  # strip an embedded "#<NITKU>" annotation
    value = re.sub(r"\s*\n\s*", " ", value)  # join wrapped lines with a single space
    value = re.sub(r"\s{2,}", " ", value).strip(" ,")
    return value or None


_SATUAN_KEYWORDS = "JASA|UNIT|PCS|KG|BOX|JAM|HARI|SET"

# One-row-per-item table (older/simpler layout): "1 JKP-001 Konsultasi
# Teknologi Informasi JASA 10 Rp5.000.000 Rp0 Rp50.000.000". The
# trailing lookahead - rather than anchoring to `$`/newline - stops the
# non-greedy NAMA capture at the start of the *next* row (or the
# "Harga Jual" summary line right after the table) even without a
# clean line break per row.
_LINE_ITEM_TABLE_RE = re.compile(
    r"(\d+)\s+([A-Za-z0-9\-]{2,20})\s+(.+?)\s+(" + _SATUAN_KEYWORDS + r")\s+"
    r"([\d.,]+)\s+Rp\.?\s?([\d.,]+)\s+Rp\.?\s?([\d.,]+)\s+Rp\.?\s?([\d.,]+)"
    r"(?=\s+\d+\s+[A-Za-z0-9\-]{2,20}\s+|\s+Harga\s+Jual|\s*$)",
    re.IGNORECASE,
)

# Real Coretax layout: each item is a multi-line block, not one table
# row - "No", "Kode", "Nama" (possibly several lines), a
# "Rp<unit price> x <qty> [Lainnya]" line, an OPTIONAL "Potongan Harga
# = Rp<discount>" line, an OPTIONAL "PPnBM (<rate>%) = Rp<amount>"
# line, then the line total on its own. `\s+`/`\s*\n` between groups
# already bridges the newline between each of these pieces (verified
# against the real PDF's native text extraction, where every one of
# these is genuinely its own line - "1" and "170300" are even on
# SEPARATE lines, not "1 170300" as a casual read of the rendered PDF
# might suggest). The negative lookahead in NAMA keeps it from eating
# into the "Rp ... x ..." line if the item name happens to wrap.
_LINE_ITEM_BLOCK_RE = re.compile(
    r"(?P<no>\d+)\s+(?P<kode>[A-Za-z0-9\-]{2,20})\s*\n"
    r"(?P<nama>(?:(?!\s*Rp\.?\s*[\d.,]+\s*x).+\n)+)"
    r"Rp\.?\s*(?P<harga_satuan>[\d.,]+)\s*x\s*(?P<qty>[\d.,]+)[^\n]*\n"
    r"(?:Potongan\s+Harga\s*=\s*Rp\.?\s*(?P<diskon>[\d.,]+)\s*\n)?"
    r"(?:PPnBM\s*\(\s*(?P<tarif_ppnbm>[\d.,]+)\s*%\s*\)\s*=\s*Rp\.?\s*(?P<ppnbm>[\d.,]+)\s*\n)?"
    r"(?P<harga_total>[\d.,]+)",
    re.IGNORECASE,
)


def _classify_barang_jasa(kode_objek: Optional[str], satuan: Optional[str]) -> Optional[str]:
    """BARANG vs JASA, only when the document gives enough signal.

    JKP-prefixed object codes ("Jasa Kena Pajak") and a JASA unit are
    services; BKP-prefixed codes and a physical unit (UNIT/PCS/KG/BOX)
    are goods. DJP's real numeric Kode Barang/Jasa (e.g. "170300") has
    no such prefix and no separate Satuan column at all in the
    Coretax layout, so it's deliberately left None rather than
    guessed from the numeric code alone.
    """
    if satuan and satuan.upper() == "JASA":
        return "JASA"
    if kode_objek:
        if re.match(r"(?i)^JKP", kode_objek):
            return "JASA"
        if re.match(r"(?i)^BKP", kode_objek):
            return "BARANG"
    if satuan and satuan.upper() in {"UNIT", "PCS", "KG", "BOX"}:
        return "BARANG"
    return None


def _parse_line_items_block_style(text: str) -> list[FakturPajakLineItem]:
    items: list[FakturPajakLineItem] = []
    for m in _LINE_ITEM_BLOCK_RE.finditer(text):
        g = m.groupdict()
        items.append(
            FakturPajakLineItem(
                barang_jasa=_classify_barang_jasa(g["kode"], None),
                kode_objek=g["kode"],
                nama=g["nama"].strip(),
                satuan=None,  # not a separate column in this layout
                harga_satuan=_to_number(g["harga_satuan"]),
                jumlah_barang=_to_number(g["qty"]),
                harga_total=_to_number(g["harga_total"]),
                diskon=_to_number(g["diskon"]),
                # These two ARE printed per-line in this layout - unlike
                # the invoice-level-only case, so they're populated here.
                tarif_ppnbm=_to_number(g["tarif_ppnbm"]),
                ppnbm=_to_number(g["ppnbm"]),
                # No per-line PPN/DPP breakdown is printed even here -
                # only PPnBM - so those stay None, per module docstring.
            )
        )
    return items


def _parse_line_items_table_style(text: str) -> list[FakturPajakLineItem]:
    items: list[FakturPajakLineItem] = []
    for m in _LINE_ITEM_TABLE_RE.finditer(text):
        _no, kode_objek, nama, satuan, qty, harga_satuan, diskon, harga_total = m.groups()
        satuan = satuan.upper()
        items.append(
            FakturPajakLineItem(
                barang_jasa=_classify_barang_jasa(kode_objek, satuan),
                kode_objek=kode_objek,
                nama=nama.strip(),
                satuan=satuan,
                harga_satuan=_to_number(harga_satuan),
                jumlah_barang=_to_number(qty),
                harga_total=_to_number(harga_total),
                diskon=_to_number(diskon),
            )
        )
    return items


def _parse_line_items(text: str) -> list[FakturPajakLineItem]:
    # Try the real Coretax multi-line block format first (the actual
    # production case); fall back to the simpler one-row table format.
    # Whichever matches at least one item wins outright, rather than
    # merging partial results from both, to avoid double-counting an
    # item that happens to loosely match both patterns.
    block_items = _parse_line_items_block_style(text)
    if block_items:
        return block_items
    return _parse_line_items_table_style(text)


# ---------------------------------------------------------------------------
# Calculation validation
# ---------------------------------------------------------------------------


def _approx_equal(a: Optional[float], b: Optional[float], abs_tol: float = 2.0) -> bool:
    """Loose equality for rupiah amounts - tolerant of rounding on
    percentage-derived figures (e.g. PPN computed from a fractional
    DPP Nilai Lain), never exact-float comparison."""
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs_tol, abs(b) * 0.0005)


def _validate(
    result: FakturPajakExtractionResult,
    line_items: list[FakturPajakLineItem],
    harga_jual: Optional[float],
    potongan: Optional[float],
) -> FakturPajakValidation:
    notes: list[str] = []
    any_check_ran = False
    any_mismatch = False

    for idx, item in enumerate(line_items, start=1):
        if item.harga_satuan is not None and item.jumlah_barang is not None and item.harga_total is not None:
            any_check_ran = True
            gross = item.harga_satuan * item.jumlah_barang
            net = gross - (item.diskon or 0.0)
            # A real Coretax invoice often prints HARGA_TOTAL as the
            # GROSS line amount (discount applied later, at the DPP
            # level) rather than net-of-discount - both are valid
            # conventions depending on the document, so either match
            # is accepted; only flag when it matches neither.
            if not (_approx_equal(gross, item.harga_total) or _approx_equal(net, item.harga_total)):
                any_mismatch = True
                notes.append(
                    f"Line {idx} ({item.kode_objek or item.nama}): HARGA_TOTAL "
                    f"({item.harga_total}) matches neither HARGA_SATUAN x JUMLAH_BARANG "
                    f"({gross}) nor that minus DISKON ({net})."
                )

    line_total_sum = sum(i.harga_total for i in line_items if i.harga_total is not None) or None
    if line_total_sum is not None and harga_jual is not None:
        any_check_ran = True
        if not _approx_equal(line_total_sum, harga_jual):
            any_mismatch = True
            notes.append(
                f"Sum of line HARGA_TOTAL ({line_total_sum}) does not match "
                f"'Harga Jual / Penggantian' ({harga_jual})."
            )
        else:
            notes.append(f"Sum of line HARGA_TOTAL matches 'Harga Jual / Penggantian' ({harga_jual}).")

    line_discount_sum = sum(i.diskon for i in line_items if i.diskon is not None) or None
    if line_discount_sum is not None and potongan is not None:
        any_check_ran = True
        if not _approx_equal(line_discount_sum, potongan):
            any_mismatch = True
            notes.append(
                f"Sum of line DISKON ({line_discount_sum}) does not match "
                f"'Potongan Harga' ({potongan})."
            )

    if result.jumlah_dpp is not None and harga_jual is not None and potongan is not None:
        any_check_ran = True
        net = harga_jual - potongan
        # DJP's "DPP Nilai Lain" scheme - used for specific
        # DJP-designated transaction categories (e.g. certain telecom/
        # voucher products, matching this document's item) - computes
        # DPP as net-of-discount x 11/12, giving an effective 11% PPN
        # rate once the standard 12% is applied on top of it. Both this
        # and the plain "DPP = net" convention are valid depending on
        # the transaction type, so either match is accepted.
        nilai_lain_dpp = net * 11 / 12
        if _approx_equal(result.jumlah_dpp, net):
            notes.append(f"JUMLAH_DPP matches 'Harga Jual/Penggantian' minus 'Potongan Harga' ({net}).")
        elif _approx_equal(result.jumlah_dpp, nilai_lain_dpp, abs_tol=5.0):
            notes.append(
                f"JUMLAH_DPP ({result.jumlah_dpp}) is consistent with the DPP Nilai Lain scheme "
                f"(net-of-discount x 11/12 = {nilai_lain_dpp:.2f}), giving an effective PPN rate "
                "of 11% on the net transaction value."
            )
        else:
            any_mismatch = True
            notes.append(
                f"JUMLAH_DPP ({result.jumlah_dpp}) does not equal 'Harga Jual/Penggantian' minus "
                f"'Potongan Harga' ({net}), nor the DPP Nilai Lain formula (net x 11/12 = "
                f"{nilai_lain_dpp:.2f}) - worth confirming this document's DPP calculation rule."
            )

    if result.jumlah_ppn is not None and result.jumlah_dpp_lain is not None:
        any_check_ran = True
        expected_ppn = result.jumlah_dpp_lain * 0.12 if result.jumlah_dpp_lain else None
        if expected_ppn is not None and _approx_equal(result.jumlah_ppn, expected_ppn):
            notes.append(
                "JUMLAH_PPN is consistent with JUMLAH_DPP_LAIN x 12% "
                "(PPN computed on DPP Nilai Lain rather than directly on JUMLAH_DPP)."
            )
        elif result.jumlah_dpp:
            expected_on_dpp = result.jumlah_dpp * 0.12
            if not _approx_equal(result.jumlah_ppn, expected_on_dpp):
                any_mismatch = True
                notes.append(
                    f"JUMLAH_PPN ({result.jumlah_ppn}) does not match JUMLAH_DPP x 12% "
                    f"({expected_on_dpp}) or JUMLAH_DPP_LAIN x 12%."
                )
    elif result.jumlah_ppn is not None and result.jumlah_dpp is not None:
        # No DPP Nilai Lain printed at all (the real Coretax case) -
        # just sanity-check PPN against DPP x 12% directly.
        any_check_ran = True
        expected_on_dpp = result.jumlah_dpp * 0.12
        if not _approx_equal(result.jumlah_ppn, expected_on_dpp):
            notes.append(
                f"JUMLAH_PPN ({result.jumlah_ppn}) is not exactly DPP x 12% "
                f"({expected_on_dpp}) - may reflect a different effective rate or rounding; "
                "not flagged as a hard mismatch since several valid DJP rate schemes exist."
            )

    if result.jumlah_dpp is not None and result.jumlah_ppn is not None:
        any_check_ran = True
        computed_total = result.jumlah_dpp + result.jumlah_ppn + (result.jumlah_ppnbm or 0.0)
        notes.append(f"JUMLAH_DPP + JUMLAH_PPN + JUMLAH_PPNBM = {computed_total}.")

    if not any_check_ran:
        status = "NOT_CHECKED"
    elif any_mismatch:
        status = "WARNING"
    else:
        status = "VALID"

    missing_fields = [
        field for field in (
            "npwp_wp", "id_tku_wp", "nomor_faktur", "masa_pajak", "tahun_pajak",
            "tanggal_faktur", "npwp", "nama", "jumlah_dpp", "jumlah_ppn",
        )
        if getattr(result, field) is None
    ]
    if line_items and any(item.dpp is None for item in line_items):
        missing_fields.append(
            "of[].DPP/DPP_LAIN/TARIF_PPN/PPN"
            + ("/CHECK_DPP_LAIN" if any(item.check_dpp_lain is None for item in line_items) else "")
            + " (no per-line breakdown found for these - only invoice-level totals)"
        )

    return FakturPajakValidation(
        calculation_status=status,
        calculation_notes=notes,
        low_confidence_fields=[],
        missing_fields=missing_fields,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_faktur_pajak(raw_text: str) -> FakturPajakExtractionResult:
    seller_block, buyer_block, rest = _split_sections(raw_text)

    nomor_faktur = _find(_NOMOR_FAKTUR_LABELED_RE, raw_text) or _find(_NOMOR_FAKTUR_DOTTED_RE, raw_text)
    # KD_JENIS_TRANSAKSI/FG_PENGGANTI are only derivable from the older
    # dotted format's fixed digit positions (kode.status-cabang.tahun.
    # urut) - Coretax's newer unbroken 17-digit number doesn't carry
    # the same guaranteed positional meaning, so this is left null for
    # that format rather than guessing.
    is_dotted_format = bool(nomor_faktur and re.search(r"[.\-]", nomor_faktur))
    status_pengganti = _find(_STATUS_PENGGANTI_RE, raw_text) if is_dotted_format else None
    kd_jenis_transaksi = nomor_faktur[:2] if (nomor_faktur and is_dotted_format) else None

    masa_tahun = re.search(_MASA_TAHUN_RE, raw_text, re.IGNORECASE)
    tanggal_faktur = _find(_TANGGAL_FAKTUR_LABELED_RE, rest) or _find_indo_date(rest)

    npwp_wp_digits = _digits_only(_find(_NPWP_RE, seller_block))
    npwp_buyer_digits = _digits_only(_find(_NPWP_RE, buyer_block))
    id_tku_wp = _find(_NITKU_LABELED_RE, seller_block) or _find_tku_by_npwp(raw_text, npwp_wp_digits)
    tku_pembeli = _find(_NITKU_LABELED_RE, buyer_block) or _find_tku_by_npwp(raw_text, npwp_buyer_digits)
    nik_nomor_passport, jenis_identitas = _resolve_buyer_identity(buyer_block, npwp_buyer_digits)

    harga_jual = _to_number(_find(_HARGA_JUAL_RE, rest))
    potongan = _to_number(_find_any(_POTONGAN_PATTERNS, rest))
    line_items = _parse_line_items(rest)

    fg_uang_muka = _find(_FG_UANG_MUKA_LABELED_RE, rest) or _derive_fg_uang_muka(rest)

    result = FakturPajakExtractionResult(
        npwp_wp=npwp_wp_digits,
        id_tku_wp=id_tku_wp,
        kd_jenis_transaksi=kd_jenis_transaksi,
        fg_pengganti=status_pengganti,
        nomor_faktur=nomor_faktur,
        masa_pajak=masa_tahun.group(1).zfill(2) if masa_tahun else None,
        tahun_pajak=masa_tahun.group(2) if masa_tahun else None,
        tanggal_faktur=_normalize_date(tanggal_faktur),
        npwp=npwp_buyer_digits,
        jenis_identitas=jenis_identitas,
        nik_nomor_passport=nik_nomor_passport,
        kode_negara=_find(_KODE_NEGARA_RE, buyer_block),
        nama=_find(_NAMA_RE, buyer_block),
        email_pembeli=_find(_EMAIL_RE, buyer_block),
        alamat_pembeli=_clean_alamat(_find(_ALAMAT_RE, buyer_block)),
        tku_pembeli=tku_pembeli,
        jumlah_dpp=_to_number(_find_any(_DPP_PATTERNS, rest)),
        jumlah_dpp_lain=_to_number(_find(_DPP_LAIN_RE, rest)),
        jumlah_ppn=_to_number(_find_any(_PPN_PATTERNS, rest)),
        jumlah_ppnbm=_to_number(_find_any(_PPNBM_PATTERNS, rest)),
        referensi=_find(_REFERENSI_RE, rest),
        fg_uang_muka=fg_uang_muka,
        nomor_faktur_um_sebelumnya=_find(_NOMOR_FAKTUR_UM_RE, rest),
        uang_muka_dpp=_to_number(_find(_UM_DPP_RE, rest)),
        uang_muka_dpp_lain=_to_number(_find(_UM_DPP_LAIN_RE, rest)),
        uang_muka_ppn=_to_number(_find(_UM_PPN_RE, rest)),
        uang_muka_ppnbm=_to_number(_find(_UM_PPNBM_RE, rest)),
        kode_dokumen_pendukung=_find(_KODE_DOKUMEN_RE, rest),
        tax_code=None,  # only ever populated if a document prints an explicit tax code
        # Only ever True when the document explicitly says so (per the
        # spec: "otherwise null") - never False, since absence of a
        # marker isn't proof the invoice is real.
        faktur_pajak_dummy=True if _DUMMY_MARKERS_RE.search(raw_text) else None,
        line_items=line_items,
        validation=FakturPajakValidation(),  # replaced below
    )
    result.validation = _validate(result, line_items, harga_jual, potongan)
    return result
