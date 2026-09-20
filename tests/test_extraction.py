"""
End-to-end tests against the synthetic sample images in samples/.

These are here to catch regressions like the deskew bug found while
building this: a preprocessing change that looks fine on a KTP-style
image can silently break sparse-text documents (it rotated an upright
document 90 degrees during initial development). Run with:

    pytest tests/ -v

Before relying on this service for anything real, replace/extend the
generated samples with actual (test) KTP scans and CompliFi document
PDFs - synthetic clean-font samples are a floor, not proof of real-world
accuracy.
"""

import pathlib

import fitz  # PyMuPDF
import numpy as np
import pytest

from app.core import ocr_engine
from app.core.preprocessing import load_pages, preprocess_for_ocr
from app.document.parser import parse_document
from app.document.signature_locator import locate_signature_box
from app.faktur_pajak.parser import parse_faktur_pajak
from app.ktp.parser import parse_ktp

SAMPLES_DIR = pathlib.Path(__file__).parent.parent / "samples"


def _ocr_sample(filename: str) -> str:
    data = (SAMPLES_DIR / filename).read_bytes()
    pages = load_pages(data, "image/png")
    return ocr_engine.image_to_text(preprocess_for_ocr(pages[0]))


def _ocr_photo_sample(filename: str) -> str:
    """Like _ocr_sample, but for a JPEG photo (rather than a flat PNG
    scan) run through the KTP pipeline's detect_card=True path."""
    data = (SAMPLES_DIR / filename).read_bytes()
    pages = load_pages(data, "image/jpeg")
    return ocr_engine.image_to_text(preprocess_for_ocr(pages[0], detect_card=True))


@pytest.mark.skipif(not (SAMPLES_DIR / "sample_ktp.png").exists(), reason="sample not generated")
def test_ktp_extraction_finds_valid_nik():
    text = _ocr_sample("sample_ktp.png")
    result = parse_ktp(text)
    assert result.nik.source == "ocr"
    assert len(result.nik.value) == 16
    assert result.nama.value == "GIFAR RAMADHAN"


def test_ktp_parser_handles_misread_colon_before_nik():
    """Regression test for a real bug: adaptive thresholding on a real
    KTP photo turned the ":" after "NIK" into a stray digit-like glyph
    that Tesseract read as a literal "2". The old NIK extractor did
    re.sub(r"\\D", "", entire_label_line) - stripping ALL non-digits -
    which fused that stray "2" directly onto the real 16-digit NIK,
    producing a bogus 17-digit value ("23171234567890123") at low
    confidence instead of the correct 16-digit one.

    This raw text is captured from the actual OCR output on that photo
    (not the image itself, to avoid bundling a third-party tutorial
    graphic in this repo) - it reproduces the exact "NIK 2 <16 digits>"
    shape that triggered the bug.
    """
    raw_text = (
        "PROVINSI DKI JAKARTA\n"
        "JAKARTA BARAT\n"
        "NIK 2 3171234567890123\n"
        "Nama : MIRA SETIAWAN\n"
        "Tempat/Tgl Lahir: JAKARTA, 18-02-1986\n"
        "Jenis Kelamin \u2014 : PEREMPUAN Gol. Darah : B\n"
        "Alamat : JL. PASTI CEPAT A7/66\n"
        "Status Perkawinan: KAWIN\n"
        "Pekerjaan : PEGAWAI SWASTA\n"
        "Kewarganegaraan : WNI JAKARTA BARAT\n"
        "Berlaku Hingga \u2014 : 22-02-2017 02-12-2012\n"
    )
    result = parse_ktp(raw_text)

    assert result.nik.value == "3171234567890123", (
        f"expected the clean 16-digit NIK, got {result.nik.value!r} - "
        "the misread colon likely fused back into the digit run"
    )
    assert result.nik.confidence >= 0.85
    assert result.jenis_kelamin.value == "PEREMPUAN"  # not bled together with "Gol. Darah : B"
    assert result.kewarganegaraan.value == "WNI"  # not bled together with the caption text
    assert result.berlaku_hingga.value == "22-02-2017"  # not the trailing unrelated date too


def test_ktp_parser_new_fields_and_agama_misread():
    """This raw text is Tesseract's ACTUAL output on the same test KTP
    photo as the test above (captured directly from a real run of this
    service's preprocessing + OCR pipeline, not hand-transcribed) - it
    covers the four newly-added fields (RT/RW, Kel/Desa, Kecamatan,
    Gol. Darah) and reproduces a real Agama bug: the colon after
    "Agama" got misread as a stray "2" fused directly onto the value
    with no space at all ("Agama 2ISLAM ie" for "Agama : ISLAM").
    Since a digit isn't a separator character, the old strip-leading-
    separators approach (correct for the "Jenis Kelamin _ : ..." case
    above) couldn't clean this one up - agama needed its own
    closed-vocabulary lookup instead, the same approach already used
    for kewarganegaraan's WNI/WNA.
    """
    raw_text = (
        "PROVINSI DKI JAKARTA\n"
        "JAKARTA BARAT\n"
        "NIK + 3171234567890123\n"
        "Nama : MIRA SETIAWAN\n"
        "Tempat/Tgl Lahir : JAKARTA, 18-02-1986\n"
        "Jenis Kelamin _ : PEREMPUAN Gol. Darah: B\n"
        "Alamat : JL. PASTI CEPAT A7/66\n"
        "RT/RW : 007/008\n"
        "Kel/Desa_ \u2014_: PEGADUNGAN.\n"
        "Kecamatan  : KALIDERES\n"
        "Agama 2ISLAM ie\n"
        "Status Perkawinan: KAWIN\n"
        "Pekerjaan : PEGAWAI SWASTA\n"
        "Kewarganegaraan : WNI JBKARTABARAT,\n"
        "Berlaku Hingga \u2014 : 22-02. 2017 02-12-2012\n"
    )
    result = parse_ktp(raw_text)

    assert result.golongan_darah.value == "B"
    assert result.rt_rw.value == "007/008"
    assert result.kel_desa.value == "PEGADUNGAN"  # not "_ \u2014_: PEGADUNGAN." with punctuation noise
    assert result.kecamatan.value == "KALIDERES"
    assert result.agama.value == "ISLAM", (
        f"expected ISLAM recovered from the misread-colon noise, got {result.agama.value!r}"
    )
    assert result.agama.confidence >= 0.8  # matched the known-vocabulary lookup, not the low-confidence fallback


def test_detect_and_crop_card_shrinks_a_held_up_photo():
    """samples/sample_ktp_held_photo.jpg is a real press photo of a
    student holding up his KTP - the card occupies roughly the left
    half of a 1200x800 frame, tilted a few degrees, with his own
    blurred face and a green wall filling the rest. Before card
    detection existed, this fed Tesseract (and this module's own
    deskew step, whose rotation estimate is derived from ALL
    non-background pixels in frame) the *entire* photo, producing
    near-total OCR garbage. This only checks the crop itself - that a
    real detection+warp happened and landed on a plausible ID-card
    shape - not OCR quality, which the end-to-end test below covers.
    """
    from app.core.preprocessing import _detect_and_crop_card

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    original = pages[0]
    cropped = _detect_and_crop_card(original)

    assert cropped.shape != original.shape, "expected a real crop, not the untouched original frame"
    crop_area = cropped.shape[0] * cropped.shape[1]
    original_area = original.shape[0] * original.shape[1]
    assert crop_area < 0.5 * original_area  # the card is roughly half the frame, not the whole photo

    aspect = max(cropped.shape[:2]) / min(cropped.shape[:2])
    assert 1.1 < aspect < 2.1  # in the neighborhood of a KTP's ~1.585 long/short ratio


@pytest.mark.skipif(not (SAMPLES_DIR / "sample_ktp_held_photo.jpg").exists(), reason="sample not present")
def test_ktp_pipeline_recovers_fields_from_a_held_up_photo():
    """End-to-end: real image bytes -> card detect/crop/upscale ->
    Tesseract -> parse_ktp, on the same real photo as the crop test
    above. Before the detect_card step (plus the fuzzy label fallback
    in ktp/parser.py, needed because the underlying photo is
    genuinely blurry enough to garble some labels too, not just
    values), this came back with every single field null. Digit-heavy
    fields (NIK, exact birth date) are still asserted as "recovered or
    not" only loosely, since no amount of preprocessing recovers digits
    that are genuinely unresolvable in a blurry source photo - the
    fields with a fixed, recognizable vocabulary (agama, kewargane-
    garaan, status_perkawinan) are where this fix is verified strictly.
    """
    text = _ocr_photo_sample("sample_ktp_held_photo.jpg")
    result = parse_ktp(text)

    assert result.agama.value == "ISLAM"
    assert result.kewarganegaraan.value == "WNI"
    assert result.status_perkawinan.value is not None and "KAWIN" in result.status_perkawinan.value
    assert result.kel_desa.value is not None and "LEPO" in result.kel_desa.value.upper()
    # At minimum, this must not regress back to "every field null".
    populated = [f for f in result.model_dump().values() if f["value"] is not None]
    assert len(populated) >= 6


@pytest.mark.skipif(not (SAMPLES_DIR / "sample_ktp_held_photo.jpg").exists(), reason="sample not present")
def test_ktp_ensemble_beats_any_single_variant_on_a_held_up_photo():
    """The full ensemble path (preprocess_for_ktp_ocr's several variants
    + merge_ktp_results, exactly what ktp_pipeline.run_ktp_pipeline now
    runs) against the same real held-up-photo sample as the tests above.

    kecamatan="BARUGA" checks the equal-confidence tie-break specifically:
    of the variants that found kecamatan at all, none of them read it
    cleanly (values seen: "@ARUGA", "t AARUGA") except one ("BARUGA") -
    the merge must surface that clean one, not just whichever ran first.

    The stronger claim this test makes is that NO SINGLE variant recovers
    both kecamatan="BARUGA" (clean) AND a usable status_perkawinan at the
    same time - one variant gets a clean kecamatan but misses
    status_perkawinan entirely, others get status_perkawinan but only a
    garbled kecamatan - while the merge gets both correct simultaneously,
    which is the actual benefit of running several preprocessing variants
    rather than committing to just one.

    (An earlier version of this test asserted the merge produces a
    strictly larger COUNT of populated fields than any single variant.
    That happened to be true only because one variant's NIK field was, at
    the time, populated by a now-removed fabrication path - see
    ktp_parser._extract_nik's Tier 4 comment - that produced a
    wrong-length "NIK" value with no real corroboration. Field count
    alone isn't a reliable signal of a good merge; field CORRECTNESS,
    checked below and via the dedicated NIK tests, is.)
    """
    from app.ktp.parser import merge_ktp_results
    from app.core.preprocessing import preprocess_for_ktp_ocr

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    variants = preprocess_for_ktp_ocr(pages[0])
    assert len(variants) > 1  # confirms detect_card found a crop and the ensemble actually ran

    per_variant = [parse_ktp(ocr_engine.image_to_text(v)) for v in variants]
    merged = merge_ktp_results(per_variant)

    assert merged.agama.value == "ISLAM"
    assert merged.kewarganegaraan.value == "WNI"
    assert merged.status_perkawinan.value is not None and "KAWIN" in merged.status_perkawinan.value
    assert merged.kecamatan.value == "BARUGA"  # the clean candidate, not "@ARUGA"/"t AARUGA"
    assert merged.berlaku_hingga.value == "SEUMUR HIDUP"

    def _has_both_clean(r) -> bool:
        return r.kecamatan.value == "BARUGA" and r.status_perkawinan.value is not None and "KAWIN" in r.status_perkawinan.value

    assert not any(_has_both_clean(r) for r in per_variant)  # no single variant gets both right...
    assert _has_both_clean(merged)  # ...but the merge does


@pytest.mark.skipif(not (SAMPLES_DIR / "sample_ktp_held_photo.jpg").exists(), reason="sample not present")
def test_ktp_deconvolution_variants_never_introduce_wrong_data():
    """The ensemble includes two Wiener-deconvolution variants aimed at
    recovering more blur-degraded fields. Tested empirically against
    this real photo, they recover NO additional fields (its blur
    doesn't match either assumed kernel shape closely enough) - but
    more importantly, they must not make anything WORSE. This is a
    real regression that was caught during development: a
    deconvolution variant's garbled rt_rw text ("gros" from a genuine
    "013/006") got assigned SOME confidence by an earlier version of
    _extract_rt_rw's fallback path, and that was enough to beat a
    different, correctly-null variant's result once merged - silently
    turning a correct "not found" into confident-looking wrong data.
    rt_rw and golongan_darah must still come back null here: neither
    is recoverable from this photo by any variant, deconvolution
    included, and the fix is that noise on these two fields must never
    receive enough confidence to be chosen over null in the first
    place - not just a lower confidence than before.
    """
    from app.ktp.parser import merge_ktp_results
    from app.core.preprocessing import preprocess_for_ktp_ocr

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    variants = preprocess_for_ktp_ocr(pages[0])
    per_variant = [parse_ktp(ocr_engine.image_to_text(v)) for v in variants]
    merged = merge_ktp_results(per_variant)

    assert merged.rt_rw.value is None
    assert merged.golongan_darah.value is None
    # And the fields the non-deconvolution variants already got right
    # must still be intact with deconvolution variants added to the mix.
    assert merged.agama.value == "ISLAM"
    assert merged.kecamatan.value == "BARUGA"


@pytest.mark.skipif(not (SAMPLES_DIR / "sample_doc.png").exists(), reason="sample not generated")
def test_document_extraction_prefers_labeled_values():
    text = _ocr_sample("sample_doc.png")
    result = parse_document(text)
    # Regression check for the deskew bug: preprocessing must not mangle
    # an upright, sparse-text document into unreadable noise.
    assert result.docvalue.value == "150.000.000"
    assert result.docvalue.confidence >= 0.8
    assert result.docdate.value == "12-09-2026"
    # Both amounts on the page should still surface as candidates even
    # though the labeled one wins the top pick.
    assert "50.000" in result.docvalue_candidates


def test_document_extraction_handles_no_matches_gracefully():
    result = parse_document("Lorem ipsum dolor sit amet, no dates or money here.")
    assert result.docdate.source == "not_found"
    assert result.docvalue.source == "not_found"
    assert result.docdate_candidates == []


# Captured from PyMuPDF's own get_text("text") output on a real
# (synthetic/test) Faktur Pajak PDF - label and value each land on
# their OWN line for this document's layout (unlike document_parser's
# sample, which has them share a line), which is exactly the case that
# exposed the "blank field bleeds into the next label" bug below.
FAKTUR_PAJAK_SAMPLE_TEXT = """FAKTUR PAJAK
Kode & Nomor Faktur
010.000-26.00001234
Pengusaha Kena Pajak
Nama
PT CONTOH TEKNOLOGI INDONESIA
NPWP
00.111.222.3-444.555
NITKU
0000000000000000000001
Alamat
Jl. Contoh Raya No. 10, Bandung, Jawa Barat 40111
Pembeli / Penerima Jasa
Nama
PT CONTOH PEMBELI NUSANTARA
Jenis Identitas
NPWP
NPWP
00.666.777.8-999.000
NIK / Nomor Paspor
Kode Negara
IDN
NITKU
0000000000000000000002
Alamat
Jl. Pembeli No. 20, Jakarta Selatan, DKI Jakarta 12560
Email
finance@contohpembeli.co.id
Tanggal Faktur
17/09/2026
Masa Pajak
09 / September 2026
Status Pengganti
0
Referensi
INV/CTI/IX/2026/0042
No
Kode Barang/Jasa
Nama Barang/Jasa
Satuan
Qty
Harga Satuan
Diskon
Harga Total
1
JKP-001
Konsultasi Teknologi Informasi
JASA
10
Rp5.000.000
Rp0
Rp50.000.000
2
JKP-002
Implementasi Sistem & Integrasi API
JASA
1
Rp35.000.000
Rp5.000.000
Rp30.000.000
3
HW-001
Perangkat Network Gateway
UNIT
2
Rp10.000.000
Rp0
Rp20.000.000
Harga Jual / Penggantian
Rp100.000.000
Potongan Harga
Rp5.000.000
DPP
Rp100.000.000
DPP Nilai Lain
Rp91.666.667
Tarif PPN
12%
PPN
Rp11.000.000
Tarif PPnBM
0%
PPnBM
Rp0
TOTAL
Rp111.000.000
Informasi Uang Muka
FG_UANG_MUKA
0 \u2014 bukan faktur uang muka
Nomor Faktur Uang Muka Sebelumnya
Uang Muka DPP
Rp0
Uang Muka DPP Nilai Lain
Rp0
Uang Muka PPN
Rp0
Uang Muka PPnBM
Rp0
Kode Dokumen Pendukung
INVOICE

Tempat, Tanggal
Bandung, 17 September 2026
Penandatangan / Otorisasi
PT CONTOH TEKNOLOGI INDONESIA
CONTOH PIC PKP
Nomor dan data pada dokumen ini adalah data contoh/fiktif untuk pengujian sistem.
"""


def test_faktur_pajak_extracts_header_and_line_items():
    result = parse_faktur_pajak(FAKTUR_PAJAK_SAMPLE_TEXT)

    assert result.npwp_wp == "001112223444555"  # seller, not buyer
    assert result.npwp == "006667778999000"  # buyer, not seller
    assert result.nomor_faktur == "010.000-26.00001234"
    assert result.kd_jenis_transaksi == "01"
    assert result.fg_pengganti == "0"
    assert result.masa_pajak == "09"
    assert result.tahun_pajak == "2026"
    assert result.tanggal_faktur == "2026-09-17"  # normalized from 17/09/2026
    assert result.nama == "PT CONTOH PEMBELI NUSANTARA"  # buyer's name, not seller's
    assert result.referensi == "INV/CTI/IX/2026/0042"
    assert result.jumlah_dpp == 100_000_000
    assert result.jumlah_dpp_lain == 91_666_667
    assert result.jumlah_ppn == 11_000_000
    assert result.faktur_pajak_dummy is True  # doc explicitly says "data contoh/fiktif"

    assert len(result.line_items) == 3
    first = result.line_items[0]
    assert first.kode_objek == "JKP-001"
    assert first.barang_jasa == "JASA"
    assert first.harga_satuan == 5_000_000
    assert first.jumlah_barang == 10
    assert first.harga_total == 50_000_000
    third = result.line_items[2]
    assert third.barang_jasa == "BARANG"  # UNIT satuan / non-JKP code
    assert third.kode_objek == "HW-001"


def test_faktur_pajak_handles_blank_optional_fields_without_bleeding_into_next_label():
    """Regression test: a non-down-payment, NPWP-identified-buyer
    invoice leaves 'NIK / Nomor Paspor' and 'Nomor Faktur Uang Muka
    Sebelumnya' blank, so those labels run straight into the NEXT
    label ('...Paspor\\nKode Negara', '...Sebelumnya\\nUang Muka DPP').
    An early version of the parser captured the next label's first
    word ('Kode', 'Uang') as if it were the value - requiring a digit
    in the captured token is what tells a real value apart from an
    accidentally-swallowed label."""
    result = parse_faktur_pajak(FAKTUR_PAJAK_SAMPLE_TEXT)
    assert result.nik_nomor_passport is None
    assert result.nomor_faktur_um_sebelumnya is None
    # And the genuinely-blank-but-adjacent label's own value is untouched.
    assert result.kode_negara == "IDN"


def test_faktur_pajak_flags_dpp_not_net_of_discount():
    """This sample's DPP (100,000,000) equals the gross 'Harga Jual /
    Penggantian' rather than that minus 'Potongan Harga' (95,000,000) -
    a real discrepancy worth surfacing, not silently accepting."""
    result = parse_faktur_pajak(FAKTUR_PAJAK_SAMPLE_TEXT)
    assert result.validation.calculation_status == "WARNING"
    assert any("does not equal" in note for note in result.validation.calculation_notes)


def test_faktur_pajak_leaves_per_line_tax_fields_null_when_only_invoice_totals_given():
    """No per-item DPP/PPN breakdown is printed on this invoice (only
    invoice-level totals) - per-line tax fields must stay None rather
    than being split pro-rata, per the explicit "no confirmed
    allocation rule yet" decision this parser was built around."""
    result = parse_faktur_pajak(FAKTUR_PAJAK_SAMPLE_TEXT)
    for item in result.line_items:
        assert item.dpp is None
        assert item.ppn is None
        assert item.check_dpp_lain is None


def test_faktur_pajak_handles_missing_document_gracefully():
    result = parse_faktur_pajak("Lorem ipsum dolor sit amet, not a faktur pajak at all.")
    assert result.nomor_faktur is None
    assert result.line_items == []
    assert result.validation.calculation_status == "NOT_CHECKED"


# Captured directly from PyMuPDF's get_text("text") on an actual (real
# production-format) DJP Coretax Faktur Pajak PDF - not hand-transcribed.
# This layout differs substantially from the simpler one above: labels
# are followed by a literal ":", numbers use "Dasar Pengenaan Pajak"/
# "Jumlah PPN" instead of "DPP"/"PPN", the invoice date only appears as
# an Indonesian-month-name date next to the e-signature, blank fields
# print a literal "-", the buyer's NITKU is embedded in their address
# as "#<22 digits>" with no label of its own, and line items are
# multi-line blocks with an embedded per-line PPnBM rate/amount rather
# than one-row-per-item.
REAL_COMPLIFI_FAKTUR_PAJAK_TEXT = """ 
Faktur Pajak
Nama: TELEKOMUNIKASI SELULAR (TELKOMSEL)
Alamat: GEDUNG TELKOM LANDMARK TOWER
MENARA 1 LT. 1-20, JL. JEND. GATOT SUBROTO
KAV. 52 , KOTA ADM. JAKARTA SELATAN
#0017183278093000000000
Kode dan Nomor Seri Faktur Pajak: 04002600002412757
Pengusaha Kena Pajak:
Nama : TELEKOMUNIKASI SELULAR (TELKOMSEL)
Alamat : GEDUNG TELKOM LANDMARK TOWER MENARA 1 LT. 1-20, JL. JEND. GATOT SUBROTO KAV. 52 ,  RT
006,  RW 001, KUNINGAN BARAT, MAMPANG PRAPATAN, KOTA ADM. JAKARTA SELATAN, DKI JAKARTA 12710
NPWP : 0017183278093000
Pembeli Barang Kena Pajak/Penerima Jasa Kena Pajak:
Nama : GOLDEN COMMUNICATION
Alamat : JL SUDIRMAN NO.216, RT 001, RW 001, TANJUNG BATU KOTA, KUNDUR, KAB. KARIMUN, KEPULAUAN
RIAU 29675 #0022767263217000000003
NPWP : 0022767263217000
NIK : -
Nomor Paspor : -
Identitas Lain : -
Email: -
No.
Kode
Barang/
Jasa
Nama Barang Kena Pajak / Jasa Kena Pajak
Harga Jual / Penggantian /
Uang Muka / Termin
(Rp)
1
170300
Voucherless Bulk Akuisisi 10Jt
Rp 9.009.009,00 x 30,00 Lainnya
Potongan Harga = Rp 21.621.630,00
PPnBM (0,00%) = Rp 0,00
270.270.270,00
Harga Jual / Penggantian / Uang Muka / Termin
270.270.270,00
Dikurangi Potongan Harga
21.621.630,00
Dikurangi Uang Muka yang telah diterima
Dasar Pengenaan Pajak
227.927.917,00
Jumlah PPN (Pajak Pertambahan Nilai)
27.351.350,00
Jumlah PPnBM (Pajak Penjualan atas Barang Mewah)
0,00
Sesuai dengan ketentuan yang berlaku, Direktorat Jenderal Pajak mengatur bahwa Faktur Pajak ini telah ditandatangani
secara elektronik sehingga tidak diperlukan tanda tangan basah pada Faktur Pajak ini.
KOTA ADM. JAKARTA SELATAN, 02 Januari
2026
Ditandatangani secara elektronik
IRVAN CHANDRA WARDHANA
(Referensi: SO-112000RG-202601-0024)
Pemberitahuan: Faktur Pajak ini telah dilaporkan ke Direktorat Jenderal Pajak dan telah memperoleh persetujuan sesuai
dengan ketentuan peraturan perpajakan yang berlaku. PERINGATAN: PKP yang membuat Faktur Pajak yang tidak sesuai
dengan keadaan yang sebenarnya dan/atau sesungguhnya sebagaimana dimaksud dalam Pasal 13 ayat (9) UU PPN
dikenai sanksi sesuai dengan Pasal 14 ayat (4) UU KUP.
1 dari 1
"""


def test_faktur_pajak_real_coretax_layout_extracts_header():
    result = parse_faktur_pajak(REAL_COMPLIFI_FAKTUR_PAJAK_TEXT)

    assert result.npwp_wp == "0017183278093000"  # seller
    assert result.npwp == "0022767263217000"  # buyer
    assert result.nomor_faktur == "04002600002412757"  # unbroken 17-digit Coretax format
    # No dotted format here, so these positional derivations must NOT fire.
    assert result.kd_jenis_transaksi is None
    assert result.fg_pengganti is None
    assert result.tanggal_faktur == "2026-01-02"  # from "...02 Januari\n2026..." near the e-signature
    assert result.nama == "GOLDEN COMMUNICATION"  # buyer's name
    assert result.referensi == "SO-112000RG-202601-0024"
    assert result.jumlah_dpp == 227_927_917
    assert result.jumlah_ppn == 27_351_350
    assert result.jumlah_ppnbm == 0


def test_faktur_pajak_real_layout_derives_nitku_from_hash_id():
    """Neither party's NITKU has its own label in this layout - each is
    embedded as "#<22 digits>" (NPWP + 6-digit branch suffix) somewhere
    in the document. Matched by digit-prefix against each party's own
    NPWP, not by position (the seller's copy appears in a "kepada"
    mailing box that comes BEFORE the seller's own labelled section)."""
    result = parse_faktur_pajak(REAL_COMPLIFI_FAKTUR_PAJAK_TEXT)
    assert result.id_tku_wp == "0017183278093000000000"
    assert result.tku_pembeli == "0022767263217000000003"


def test_faktur_pajak_real_layout_handles_dash_as_blank_and_derives_jenis_identitas():
    """NIK/Nomor Paspor/Identitas Lain are all printed as a literal "-"
    on this real invoice (buyer identified by NPWP instead) - and this
    layout has no explicit "Jenis Identitas" label at all, so it must
    be DERIVED from which identity field is actually populated."""
    result = parse_faktur_pajak(REAL_COMPLIFI_FAKTUR_PAJAK_TEXT)
    assert result.nik_nomor_passport is None
    assert result.jenis_identitas == "NPWP"


def test_faktur_pajak_real_layout_joins_wrapped_address_and_strips_hash_id():
    """The buyer's address wraps across two physical lines AND has the
    "#<NITKU>" annotation appended directly to it with no separator -
    both need cleaning up rather than leaking into alamat_pembeli."""
    result = parse_faktur_pajak(REAL_COMPLIFI_FAKTUR_PAJAK_TEXT)
    assert result.alamat_pembeli == (
        "JL SUDIRMAN NO.216, RT 001, RW 001, TANJUNG BATU KOTA, KUNDUR, "
        "KAB. KARIMUN, KEPULAUAN RIAU 29675"
    )
    assert "#" not in result.alamat_pembeli


def test_faktur_pajak_real_layout_derives_fg_uang_muka_from_blank_deduction_line():
    """No explicit FG_UANG_MUKA field exists in this layout - only
    "Dikurangi Uang Muka yang telah diterima", printed with no amount
    at all when there's no down payment. The 0/1 flag is derived from
    whether that line has a value next to it, not invented."""
    result = parse_faktur_pajak(REAL_COMPLIFI_FAKTUR_PAJAK_TEXT)
    assert result.fg_uang_muka == "0"


def test_faktur_pajak_real_layout_line_item_and_dpp_nilai_lain_validation():
    """This document's per-line HARGA_TOTAL is GROSS (not net of the
    per-line discount), and its DPP follows the "DPP Nilai Lain"
    scheme (net-of-discount x 11/12, giving an effective 11% PPN rate)
    rather than a straight net-of-discount figure. Both are valid,
    real DJP conventions the validator must recognize rather than
    flagging as a mismatch."""
    result = parse_faktur_pajak(REAL_COMPLIFI_FAKTUR_PAJAK_TEXT)
    assert len(result.line_items) == 1
    item = result.line_items[0]
    assert item.kode_objek == "170300"
    assert item.harga_satuan == 9_009_009
    assert item.jumlah_barang == 30
    assert item.harga_total == 270_270_270  # gross, not net-of-discount
    assert item.diskon == 21_621_630
    assert item.tarif_ppnbm == 0  # per-line PPnBM IS printed in this layout
    assert item.ppnbm == 0

    assert result.validation.calculation_status == "VALID"
    assert any("DPP Nilai Lain" in note for note in result.validation.calculation_notes)


def _build_signature_test_pdf() -> bytes:
    """A minimal synthetic PDF mimicking the exact structure that broke
    the first version of signature_locator: the signer's FULL name
    appears once as a header identity field, and again - ABBREVIATED -
    under the real signature block, inside a two-column ruled table
    with a blank cell above the printed name."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)

    # Decoy: an identity field near the top, with the UNABBREVIATED
    # name - this is what a naive "highest text-similarity" match would
    # pick, and is NOT where a signature should be placed.
    page.insert_text((20, 30), "Nama: Muhammad Gifar Zaini", fontsize=10)

    # Signature table: two columns ("Pemohon" / "Approver"), each with
    # a header row, a blank row (the signature space), and a name row.
    left, mid, right = 20, 200, 380
    top, header_bot, blank_bot, bottom = 200, 215, 260, 275
    shape = page.new_shape()
    for y in (top, header_bot, blank_bot, bottom):
        shape.draw_line((left, y), (right, y))
    for x in (left, mid, right):
        shape.draw_line((x, top), (x, bottom))
    shape.finish(width=0.5, color=(0, 0, 0))
    shape.commit()

    page.insert_text((left + 5, top + 11), "Pemohon", fontsize=9)
    page.insert_text((mid + 5, top + 11), "Approver", fontsize=9)
    # The real target: an ABBREVIATED name under the blank signature cell.
    page.insert_text((left + 5, bottom - 5), "Muhammad Gifar Z.", fontsize=9)
    page.insert_text((mid + 5, bottom - 5), "Someone Else", fontsize=9)

    return doc.tobytes()


def test_signature_locator_prefers_signature_block_over_header_field():
    """Regression test: a plain text-similarity match picks the
    unabbreviated header field over the abbreviated signature-block
    name, since it scores a perfect 1.0. The locator must instead
    prefer whichever candidate sits in a genuinely ruled table cell."""
    pdf_bytes = _build_signature_test_pdf()
    matches = locate_signature_box(pdf_bytes, "application/pdf", "Muhammad Gifar Zaini")

    assert len(matches) == 1
    match = matches[0]
    assert match.matched_text == "Muhammad Gifar Z."
    assert match.method == "pdf_native"
    # The blank cell sits between the header row and the name row, in
    # the "Pemohon" (left) column - not the header field's location.
    assert 15 <= match.x0 <= 25
    assert 195 <= match.x1 <= 205
    assert 214 <= match.y0 <= 216
    assert 259 <= match.y1 <= 261


def test_signature_locator_returns_no_match_for_absent_signer():
    pdf_bytes = _build_signature_test_pdf()
    matches = locate_signature_box(pdf_bytes, "application/pdf", "Someone Not On This Document")
    assert matches == []


def test_signature_locator_resolves_correct_column_for_second_signer():
    pdf_bytes = _build_signature_test_pdf()
    matches = locate_signature_box(pdf_bytes, "application/pdf", "Someone Else")
    assert len(matches) == 1
    assert matches[0].matched_text == "Someone Else"
    assert matches[0].x0 > 195  # right-hand ("Approver") column, not the left one


# ---------------------------------------------------------------------------
# NIK region cross-check (ktp/field_ocr.py + ktp/parser.reconcile_nik)
# ---------------------------------------------------------------------------


def test_normalize_ocr_digits_maps_common_letter_confusions():
    from app.ktp.parser import _normalize_ocr_digits

    assert _normalize_ocr_digits("O1IlS8B") == "0111588"
    assert _normalize_ocr_digits("3273010101900001") == "3273010101900001"  # already-clean digits untouched


def test_extract_nik_tier4_recovers_value_via_letter_digit_normalization():
    """A real failure mode this is a regression test for: OCR reading a
    couple of NIK digits as similar-looking letters ("O" for "0", "I" for
    "1"). The old last-resort tier stripped non-digit characters OUTRIGHT
    (re.sub(r"\\D", "", ...)), which deleted those letters rather than
    correcting them - silently shortening a genuinely-16-digit NIK to 14
    and, with the old off-by-one tolerance, could still slip through at
    low confidence. Now the letters are corrected to their digit reading
    FIRST (_normalize_ocr_digits), so the strip-non-digits pass has
    nothing left to accidentally delete.
    """
    from app.ktp.parser import _extract_nik

    raw_text = "NIK : 3I7I0301O19OOOO1\nNama : BUDI SANTOSO\n"
    result = _extract_nik([ln for ln in raw_text.splitlines() if ln.strip()])
    assert result.value == "3171030101900001"
    assert result.confidence > 0


def test_extract_nik_no_longer_fabricates_off_by_one_length_value():
    """The removed Tier 4 fallback used to accept a 15- or 17-digit
    result at low confidence rather than nothing at all - exactly the
    kind of fabricated-looking value the extraction spec this service was
    built against explicitly warns against ("do not fabricate OCR
    values... return null rather than inventing a number"). A label line
    whose digit run - even after letter/digit normalization - comes out
    to the wrong length must now return not_found, full stop; recovery
    from here on only happens via an independent corroborating signal
    (reconcile_nik), not by relaxing this function's own tolerance.
    """
    from app.ktp.parser import _extract_nik

    raw_text = "NIK : 317103010190000\nNama : BUDI SANTOSO\n"  # 15 digits, no letters to fix
    result = _extract_nik([ln for ln in raw_text.splitlines() if ln.strip()])
    assert result.value is None
    assert result.source == "not_found"


def test_reconcile_nik_agreement_yields_high_confidence():
    from app.ktp.parser import reconcile_nik
    from app.schemas import FieldResult

    text_result = FieldResult(value="3171030101900001", confidence=0.9, source="ocr")
    reconciled = reconcile_nik(text_result, ["3171030101900001", "3171030101900001"])
    assert reconciled.value == "3171030101900001"
    assert reconciled.confidence >= 0.95


def test_reconcile_nik_close_disagreement_lowers_confidence_but_keeps_text_value():
    from app.ktp.parser import reconcile_nik
    from app.schemas import FieldResult

    text_result = FieldResult(value="3171030101900001", confidence=0.9, source="ocr")
    # One digit off from the text-tier value (a plausible single-glyph misread).
    reconciled = reconcile_nik(text_result, ["3171030101900002"])
    assert reconciled.value == "3171030101900001"
    assert reconciled.confidence < text_result.confidence


def test_reconcile_nik_wide_disagreement_returns_null_rather_than_guessing():
    from app.ktp.parser import reconcile_nik
    from app.schemas import FieldResult

    text_result = FieldResult(value="3171030101900001", confidence=0.9, source="ocr")
    # A completely different 16-digit number - not a plausible single-digit misread.
    reconciled = reconcile_nik(text_result, ["9998887776665554"])
    assert reconciled.value is None
    assert reconciled.source == "not_found"


def test_reconcile_nik_region_only_recovers_when_whole_text_found_nothing():
    from app.ktp.parser import reconcile_nik
    from app.schemas import FieldResult

    text_result = FieldResult()  # whole-text extraction found nothing usable
    reconciled = reconcile_nik(text_result, ["3171030101900001", "3171030101900001", "9999999999999999"])
    assert reconciled.value == "3171030101900001"  # the majority candidate
    assert 0 < reconciled.confidence < 0.9  # capped below a text+region agreement


def test_reconcile_nik_returns_null_when_nothing_found_anywhere():
    from app.ktp.parser import reconcile_nik
    from app.schemas import FieldResult

    reconciled = reconcile_nik(FieldResult(), [])
    assert reconciled.value is None
    assert reconciled.source == "not_found"


def test_locate_nik_value_crop_finds_region_next_to_real_label():
    """Runs field_ocr.locate_nik_value_crop against the real word boxes
    Tesseract produces for the bundled flat-scan KTP sample, and checks
    that a second, independent digits-only OCR pass over the returned
    crop recovers the same 16-digit NIK as the whole-text parser."""
    from app.ktp.field_ocr import locate_nik_value_crop
    from app.ktp.parser import parse_ktp

    data = (SAMPLES_DIR / "sample_ktp.png").read_bytes()
    pages = load_pages(data, "image/png")
    processed = preprocess_for_ocr(pages[0])

    text_result = parse_ktp(ocr_engine.image_to_text(processed))
    assert text_result.nik.value is not None

    words = ocr_engine.image_to_data(processed)
    crop = locate_nik_value_crop(processed, words)
    assert crop is not None

    digits, _conf = ocr_engine.image_to_digits(crop)
    assert digits == text_result.nik.value


def test_locate_nik_value_crop_returns_none_without_a_label():
    from app.ktp.field_ocr import locate_nik_value_crop

    assert locate_nik_value_crop(np.zeros((10, 10), dtype="uint8"), []) is None


# ---------------------------------------------------------------------------
# Image quality assessment (ktp/image_quality.py)
# ---------------------------------------------------------------------------


def test_assess_quality_flags_a_genuinely_blurry_photo():
    from app.ktp.image_quality import assess_quality
    from app.core.preprocessing import detect_and_crop_card, preprocess_for_ktp_ocr

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    cropped = detect_and_crop_card(pages[0])
    variants = preprocess_for_ktp_ocr(pages[0])

    quality = assess_quality(pages[0], cropped, variants[0])
    assert quality.document_detected is True
    assert quality.blur_score < 0.3  # this sample is a genuinely soft/blurry photo
    assert quality.document_area_ratio is not None and quality.document_area_ratio < 1.0


def test_assess_quality_reports_no_detection_for_a_flat_scan():
    from app.ktp.image_quality import assess_quality
    from app.core.preprocessing import preprocess_for_ocr

    data = (SAMPLES_DIR / "sample_ktp.png").read_bytes()
    pages = load_pages(data, "image/png")
    quality = assess_quality(pages[0], None, preprocess_for_ocr(pages[0]))
    assert quality.document_detected is False
    assert quality.perspective_corrected is False
    assert quality.document_area_ratio is None


def test_assess_quality_does_not_flag_a_plain_white_background_as_glare():
    """Regression test: an earlier version of the glare heuristic looked
    only at the RAW FRACTION of near-saturated pixels, which flagged any
    mostly-white page (like this service's own synthetic test samples, or
    a legitimately bright card background) as "glare" even with zero
    actual reflection on it. Real specular glare is a spatially
    concentrated hotspot, not a uniformly bright frame - see
    image_quality._glare_detected.
    """
    from app.ktp.image_quality import assess_quality
    from app.core.preprocessing import preprocess_for_ocr

    data = (SAMPLES_DIR / "sample_ktp.png").read_bytes()
    pages = load_pages(data, "image/png")
    quality = assess_quality(pages[0], None, preprocess_for_ocr(pages[0]))
    assert quality.glare_detected is False


# ---------------------------------------------------------------------------
# End-to-end pipeline (ktp/pipeline.py)
# ---------------------------------------------------------------------------


def test_ktp_pipeline_nik_never_returns_an_invalid_length_value():
    """The core regression test for the bug this round of changes was
    built to fix: against the real held-up-photo sample, the OLD pipeline
    returned a fabricated 15-digit "NIK" at low (0.3) confidence - present
    enough to look like a usable value, wrong enough to be actively
    misleading if a caller relied on "value is not None" rather than
    checking confidence/length itself. The new pipeline must return
    EITHER a properly-validated 16-digit value, OR null - never anything
    in between.
    """
    from app.ktp.pipeline import run_ktp_pipeline

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    pipeline = run_ktp_pipeline(pages)

    nik = pipeline.result.nik
    assert nik.value is None or (len(nik.value) == 16 and nik.value.isdigit())


def test_ktp_pipeline_recovers_high_confidence_nik_on_a_clean_scan():
    from app.ktp.pipeline import run_ktp_pipeline

    data = (SAMPLES_DIR / "sample_ktp.png").read_bytes()
    pages = load_pages(data, "image/png")
    pipeline = run_ktp_pipeline(pages)

    assert pipeline.result.nik.value == "3273010101900001"
    assert pipeline.result.nik.confidence >= 0.9  # whole-text and region reads agree
    assert pipeline.quality is not None


def test_ktp_pipeline_surfaces_blur_warning_for_the_held_up_photo():
    from app.ktp.pipeline import run_ktp_pipeline

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    pipeline = run_ktp_pipeline(pages)

    assert any("blurry" in w for w in pipeline.warnings)


# ---------------------------------------------------------------------------
# Card-detection rectangularity check (preprocessing._detect_and_crop_card)
# ---------------------------------------------------------------------------


def test_detect_and_crop_card_rejects_a_non_rectangular_distractor():
    """Regression test for a real bug: on a clean, flat, studio-style KTP
    template photo (sample_ktp_flat_template.jpg), the card's own outer
    edge is a soft, low-contrast boundary against a similarly-toned plain
    background, which Canny + the morphological closing in
    _detect_and_crop_card never joined into one closed contour. The
    largest contour actually found was an unrelated internal watermark
    graphic - a heavily SHEARED quadrilateral that nonetheless happened
    to clear the area-share and card aspect-ratio checks by coincidence,
    producing a perspective warp that sheared every field's label out of
    line with its own value (confirmed by inspecting the crop: each
    label ended up sitting next to the WRONG value, off by one row).

    detect_and_crop_card must now reject that candidate on
    rectangularity grounds (see _MAX_CORNER_ANGLE_DEVIATION) and fall
    back to the untouched image instead of confidently returning a
    corrupted crop.
    """
    from app.core.preprocessing import detect_and_crop_card

    data = (SAMPLES_DIR / "sample_ktp_flat_template.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    cropped = detect_and_crop_card(pages[0])
    assert cropped.shape == pages[0].shape  # falls back to the unchanged image, not a bad crop


def test_detect_and_crop_card_still_finds_a_real_held_up_card():
    """The rectangularity check added alongside the regression test above
    must not itself become a false-positive machine - the genuine,
    correctly-perspective-warped detection on the real held-up-photo
    sample (already relied on by several other tests in this file) has
    to keep working."""
    from app.core.preprocessing import detect_and_crop_card

    data = (SAMPLES_DIR / "sample_ktp_held_photo.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    cropped = detect_and_crop_card(pages[0])
    assert cropped.shape != pages[0].shape  # a real crop DID happen
    assert cropped.shape[0] < pages[0].shape[0]  # and it's smaller, i.e. an actual crop


def test_ktp_pipeline_extracts_all_fields_cleanly_from_the_flat_template():
    """End-to-end coverage for the same photo the rectangularity fix
    above targets, plus two smaller bugs surfaced by debugging it:

    1. field_ocr's NIK region OCR occasionally misreads a crop's last
       digit at psm 7 ("single line") while psm 8 ("single word") on the
       exact same crop gets it right (or vice versa) - the pipeline now
       tries both and lets reconcile_nik's own agreement handling settle
       it (see ktp_pipeline.run_ktp_pipeline's comment), rather than
       trusting a single psm choice.
    2. _clean_value's separator-character class used to not include a
       plain apostrophe - so a fuzzy-matched label tail starting with
       one (e.g. "'\u2014 : PEGADUNGAN", from an OCR misread of "Kel/
       Desa" as "KellDesa" with no real "/") came back with that stray
       prefix still attached instead of being stripped down to
       "PEGADUNGAN" (see test_clean_value_strips_apostrophe_prefixed_
       stray_punctuation for the narrower unit test of the same fix).
    """
    from app.ktp.pipeline import run_ktp_pipeline

    data = (SAMPLES_DIR / "sample_ktp_flat_template.jpg").read_bytes()
    pages = load_pages(data, "image/jpeg")
    pipeline = run_ktp_pipeline(pages)
    r = pipeline.result

    assert r.nik.value == "3171234567890123"
    assert r.nik.confidence >= 0.9  # whole-text and both region passes agree
    assert r.nama.value == "MIRA SETIAWAN"
    assert r.tempat_lahir.value == "JAKARTA"
    assert r.tanggal_lahir.value == "18-02-1986"
    assert r.jenis_kelamin.value == "PEREMPUAN"
    assert r.golongan_darah.value == "B"
    assert r.rt_rw.value == "007/008"
    assert r.kel_desa.value == "PEGADUNGAN"  # not "'\u2014 : PEGADUNGAN" - see docstring
    assert r.kecamatan.value == "KALIDERES"
    assert r.agama.value == "ISLAM"
    assert r.status_perkawinan.value == "KAWIN"
    assert r.pekerjaan.value == "PEGAWAI SWASTA"
    assert r.kewarganegaraan.value == "WNI"
    assert r.berlaku_hingga.value == "22-02-2017"


# ---------------------------------------------------------------------------
# _clean_value stray-punctuation stripping (ktp/parser.py)
# ---------------------------------------------------------------------------


def test_clean_value_strips_apostrophe_prefixed_stray_punctuation():
    from app.ktp.parser import _clean_value

    assert _clean_value("'\u2014 : PEGADUNGAN") == "PEGADUNGAN"
    assert _clean_value("  : BUDI SANTOSO  ") == "BUDI SANTOSO"  # ordinary case still works
    assert _clean_value("\u2018\u2019 KALIDERES") == "KALIDERES"  # curly quotes too
