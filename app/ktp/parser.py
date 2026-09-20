"""
Parses raw OCR text from an Indonesian KTP (national ID card) into
structured fields.

Approach: KTPs have a fixed label layout ("NIK", "Nama", "Tempat/Tgl
Lahir", ...) printed on the left, with the value either on the same line
(after a colon) or occasionally wrapping to the next line. We scan
line-by-line for each label and pull the value that follows it, rather
than trying to do full free-form NLP - this is far more robust against
Tesseract's line-ordering quirks than a single regex over the whole
blob.

This is a starting point, not a finished parser: tune LABEL_PATTERNS and
the NIK-recovery regex against a batch of real (test) KTP photos before
relying on this for anything other than "pre-fill and let the user
confirm".
"""

from __future__ import annotations

import re

from app.schemas import FieldResult, KTPExtractionResult

# Maps each output field to the label variants Tesseract tends to produce
# for it (OCR mangles "Kewarganegaraan" -> "Kewarganegaraan"/"Kewarganegaraah"
# etc., so each list has a couple of common misreads).
LABEL_PATTERNS: dict[str, list[str]] = {
    "nik": [r"NIK"],
    "nama": [r"Nama"],
    "tempat_tgl_lahir": [r"Tempat[/\s]*Tgl\s*Lahir", r"Tempat.{0,3}Lahir"],
    "jenis_kelamin": [r"Jenis\s*Kelamin"],
    "golongan_darah": [r"Gol\.?\s*Darah"],
    "alamat": [r"Alamat"],
    "rt_rw": [r"RT\s*/\s*RW"],
    "kel_desa": [r"Kel(?:urahan)?\s*/\s*Desa"],
    "kecamatan": [r"Kecamatan"],
    "agama": [r"Agama"],
    "status_perkawinan": [r"Status\s*Perkawinan"],
    "pekerjaan": [r"Pekerjaan"],
    "kewarganegaraan": [r"Kewarganegaraan"],
    "berlaku_hingga": [r"Berlaku\s*Hingga"],
}

NIK_DIGITS = 16

# "12-05-1990", "12/05/1990", "12 05 1990" style dates that often trail
# the birthplace on the "Tempat/Tgl Lahir" line, e.g. "BANDUNG, 12-05-1990".
DATE_RE = re.compile(r"\b(\d{1,2})[\-/\.\s](\d{1,2})[\-/\.\s](\d{4})\b")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


# Normalized (letters only, uppercase, no spaces) form of each field's
# label - used ONLY by the fuzzy fallback below, as a last resort when
# no line matched the exact LABEL_PATTERNS regex ANYWHERE in the
# document. A blurry/low-resolution photo (e.g. a card held up at a
# distance rather than flat-scanned) can garble the LABEL itself, not
# just the value after it - an inserted space splitting "Agama" into
# "Aga ma", a misread letter turning "Perkawinan" into "Pemawnan" -
# which the exact regex has no way to recognize. "nik"/"nama" aren't
# here: NIK already has its own multi-tier digit-pattern recovery, and
# "NAMA" is short enough (4 letters) that fuzzy-matching it against
# arbitrary noisy lines risks more false positives than it recovers.
_FUZZY_LABELS: dict[str, str] = {
    "tempat_tgl_lahir": "TEMPATTGLLAHIR",
    "jenis_kelamin": "JENISKELAMIN",
    "golongan_darah": "GOLDARAH",
    "alamat": "ALAMAT",
    "rt_rw": "RTRW",
    "kel_desa": "KELDESA",
    "kecamatan": "KECAMATAN",
    "agama": "AGAMA",
    "status_perkawinan": "STATUSPERKAWINAN",
    "pekerjaan": "PEKERJAAN",
    "kewarganegaraan": "KEWARGANEGARAAN",
    "berlaku_hingga": "BERLAKUHINGGA",
}


def _exact_claimed_lines(lines: list[str]) -> set[int]:
    """Line indices that already gave a clean, exact regex match for
    SOME field - excluded from every OTHER field's fuzzy fallback
    below. Otherwise a correctly-spelled "Alamat" line (an easy
    near-miss for the "Agama" target - "ALAMA", the first 5 letters,
    is only 1 edit from "AGAMA") could get mis-claimed by a different
    field's fuzzy search just because it happens to be close enough,
    even though it's already confidently spoken for."""
    claimed: set[int] = set()
    for patterns in LABEL_PATTERNS.values():
        for idx, line in enumerate(lines):
            if idx in claimed:
                continue
            if any(re.search(pat, line, flags=re.IGNORECASE) for pat in patterns):
                claimed.add(idx)
    return claimed


def _find_label_line_fuzzy(lines: list[str], field_key: str) -> tuple[int, str] | None:
    """Edit-distance fallback for a label the exact regex couldn't find
    anywhere. Walks each line's letters (ignoring spaces/punctuation,
    which is exactly what OCR tends to mangle in a label) while keeping
    track of which RAW character position each of those letters came
    from - the value can then be cut starting from a real position in
    the original (still-punctuated, still-spaced) line, whether or not
    the line has a ":" to split on (a very garbled line - e.g. "Aga ma
    ISLAM" - often doesn't).

    Tolerance scales with label length so short labels stay strict
    (matching "ALAMAT" within 1 edit is meaningfully selective; the
    same tolerance on a 17-letter label would barely constrain
    anything) - a couple of OCR slips is normal, more than that on a
    short label usually means a real mismatch rather than noise. Lines
    already claimed by a different field's EXACT match are skipped
    entirely (see _exact_claimed_lines).
    """
    target = _FUZZY_LABELS.get(field_key)
    if target is None:
        return None
    max_distance = 1 if len(target) <= 6 else (2 if len(target) <= 12 else 3)
    claimed = _exact_claimed_lines(lines)

    for idx, line in enumerate(lines):
        if idx in claimed:
            continue
        raw_window = line[: len(target) + 12]
        norm_chars: list[str] = []
        raw_positions: list[int] = []
        for i, ch in enumerate(raw_window):
            if ch.isalpha():
                norm_chars.append(ch.upper())
                raw_positions.append(i)
        if not norm_chars:
            continue
        normalized = "".join(norm_chars)

        best: tuple[int, int] | None = None  # (distance, raw_cut_index)
        min_len = max(1, len(target) - 3)
        max_len = min(len(normalized), len(target) + 3)
        for length in range(min_len, max_len + 1):
            distance = _levenshtein(normalized[:length], target)
            if best is None or distance < best[0]:
                best = (distance, raw_positions[length - 1] + 1)
        if best is not None and best[0] <= max_distance:
            return idx, line[best[1]:].lstrip(" :.-")
    return None


def _find_label_line(lines: list[str], field_key: str) -> tuple[int, str] | None:
    for idx, line in enumerate(lines):
        for pat in LABEL_PATTERNS.get(field_key, []):
            m = re.search(pat, line, flags=re.IGNORECASE)
            if m:
                return idx, line[m.end():].lstrip(" :.-")
    # Only tried when NOTHING matched the exact pattern anywhere in the
    # document - deliberately not interleaved per-line with the exact
    # pass, so a clean document's matching is unaffected by this at all.
    return _find_label_line_fuzzy(lines, field_key)


def _clean_value(raw: str) -> str:
    # Strip any leading/trailing RUN of separator-ish characters - not
    # just repeats of one character. OCR often renders a single real
    # ":" as more than one stray glyph in a row (a faint rule or
    # underline near the colon reads as "_", a misread colon itself as
    # "+" or a stray digit) - e.g. "Jenis Kelamin _ : PEREMPUAN" for a
    # card that only has one real ":". Stripping a mixed-character run
    # from each end (rather than .strip() with a fixed char set, which
    # stops at the first character not in that set) removes all of it
    # in one pass instead of leaving "_ : PEREMPUAN" behind.
    #
    # The character class covers every separator-ish glyph seen in
    # practice, not just ":" and "-" - notably including the plain
    # apostrophe/backtick/quote family. A regex anchored at ^ or $ stops
    # dead at the FIRST character outside its class, so leaving even one
    # of these out doesn't just fail to strip that one character - it
    # blocks the whole run from being stripped at all (e.g. a fuzzy-
    # matched label's tail landing as "'\u2014 : PEGADUNGAN": since the
    # class used to lack "'", nothing was stripped and the value came
    # back as "'\u2014 : PEGADUNGAN" instead of "PEGADUNGAN" - a real
    # bug found via a real photo, see
    # test_clean_value_strips_apostrophe_prefixed_stray_punctuation).
    separator_chars = r"\s:.\-_'`\",;|~\u2014\u2013\u2018\u2019\u201c\u201d"
    value = re.sub(rf"^[{separator_chars}]+", "", raw)
    value = re.sub(rf"[{separator_chars}]+$", "", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value


# Common single-glyph OCR confusions specific to a DIGITS-only field like
# NIK - a misread that swaps a digit for a similar-looking LETTER. Used
# ONLY inside _extract_nik's lossy last-resort tiers below, where the tail
# is otherwise unstructured OCR text that may contain such letters mixed
# in with real digits: applying this BEFORE stripping non-digit characters
# turns e.g. "O" into "0" instead of just deleting it outright, which
# previously shortened an otherwise-correctly-read 16-digit NIK to 15 once
# the letter was stripped rather than corrected (a real failure mode, not
# hypothetical - see reconcile_nik / tests/test_extraction.py). Never
# applied to Tier 1/2 below, which only ever accept an already-contiguous,
# unambiguous \d{16} run with nothing to correct.
_DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8"})


def _normalize_ocr_digits(text: str) -> str:
    return text.translate(_DIGIT_CONFUSIONS)


def _extract_nik(lines: list[str]) -> FieldResult:
    hit = _find_label_line(lines, "nik")
    label_tail = hit[1] if hit is not None else ""

    # Tier 1: a clean, already-contiguous 16-digit run right on the
    # label line - the most trustworthy signal, since nothing (like a
    # misread colon) intruded between digits. Searching the raw text
    # (not space-stripped) matters: "NIK 2 3171234567890123" - where
    # a misread ":" became a stray "2" - still has a real space
    # between that "2" and the true NIK, so this finds the correct
    # 16-digit run directly instead of merging the two.
    m = re.search(r"\d{16}", label_tail)
    if m:
        return FieldResult(value=m.group(0), confidence=0.9, source="ocr")

    # Tier 2: a clean 16-digit run anywhere else on the card - covers
    # the "NIK" label itself being misread but the digit string
    # surviving intact elsewhere.
    for line in lines:
        m = re.search(r"\d{16}", line)
        if m:
            return FieldResult(value=m.group(0), confidence=0.85, source="ocr")

    # Tier 3: digits split only by internal whitespace on the label
    # line (some renders visually group the NIK in blocks, e.g.
    # "3171 2345 6789 0123").
    collapsed = re.sub(r"(?<=\d)\s+(?=\d)", "", label_tail)
    m = re.search(r"\d{16}", collapsed)
    if m:
        return FieldResult(value=m.group(0), confidence=0.7, source="ocr")

    # Tier 4: last resort - normalize common digit/letter confusions
    # (see _normalize_ocr_digits) THEN strip every remaining non-digit
    # character from the label line, accepting the result only if it
    # comes out to EXACTLY 16 digits.
    #
    # An earlier version of this tier also accepted an off-by-one length
    # (15 or 17 digits) at low confidence, on the theory that a "close
    # enough" candidate is better than nothing. In practice that produced
    # exactly the kind of fabricated-looking value this service is meant
    # to avoid: a value that fails NIK's own strict format rule (exactly
    # 16 digits) has no legitimate reading as a NIK at all - reporting one
    # anyway, even at low confidence, risks a caller treating "some value
    # is present" as "there's a real NIK here to review" rather than
    # "unreadable". Recovering a plausible value when this tier's own
    # digit run comes out short/long now happens ONLY via an independent
    # corroborating signal - see reconcile_nik, which combines this
    # function's result with a second, region-cropped digit-only OCR pass
    # (field_ocr.locate_nik_value_crop) rather than trusting either read
    # alone.
    digits = re.sub(r"\D", "", _normalize_ocr_digits(label_tail))
    if len(digits) == NIK_DIGITS:
        return FieldResult(value=digits, confidence=0.6, source="ocr")

    return FieldResult()


def _extract_simple(
    lines: list[str], field_key: str, confidence: float = 0.75, cut_before: str | None = None
) -> FieldResult:
    """cut_before: if given, truncates the value right before this pattern
    - handles KTP layouts where a second field shares the physical line
    (e.g. "PEREMPUAN Gol. Darah : B" - Jenis Kelamin and Gol. Darah are
    two separate fields printed on one line)."""
    hit = _find_label_line(lines, field_key)
    if hit is None:
        return FieldResult()
    value = _clean_value(hit[1])
    if not value and hit[0] + 1 < len(lines):
        # Some layouts put the value on the next line entirely.
        value = _clean_value(lines[hit[0] + 1])
    if cut_before:
        m = re.search(cut_before, value, flags=re.IGNORECASE)
        if m:
            value = _clean_value(value[: m.start()])
    if not value:
        return FieldResult()
    return FieldResult(value=value, confidence=confidence, source="ocr")


def _extract_tempat_tanggal_lahir(lines: list[str]) -> tuple[FieldResult, FieldResult]:
    hit = _find_label_line(lines, "tempat_tgl_lahir")
    if hit is None:
        return FieldResult(), FieldResult()

    value = _clean_value(hit[1])
    date_match = DATE_RE.search(value)
    if date_match:
        tempat = _clean_value(value[: date_match.start()].rstrip(" ,"))
        tanggal = date_match.group(0)
        return (
            FieldResult(value=tempat or None, confidence=0.7 if tempat else 0.0, source="ocr" if tempat else "not_found"),
            FieldResult(value=tanggal, confidence=0.85, source="ocr"),
        )
    # No date on this line - treat the whole thing as birthplace only.
    return FieldResult(value=value, confidence=0.6, source="ocr") if value else FieldResult(), FieldResult()


def _extract_berlaku_hingga(lines: list[str]) -> FieldResult:
    """Berlaku Hingga is either a date or the literal "SEUMUR HIDUP"
    (lifetime). Unlike other free-text fields, we know exactly what
    shape a valid value takes here, so - unlike jenis_kelamin/agama/etc,
    which just take whatever text follows the label - extract just the
    matching date (or "SEUMUR HIDUP") rather than the raw tail. This
    also sidesteps the two-column bleed-through seen on this card,
    where an unrelated photo-caption date on the same physical line
    ("Berlaku Hingga : 22-02-2017  02-12-2012") would otherwise get
    appended onto the real value.
    """
    hit = _find_label_line(lines, "berlaku_hingga")
    if hit is None:
        return FieldResult()
    value = _clean_value(hit[1])
    if re.search(r"seumur\s*hidup", value, flags=re.IGNORECASE):
        return FieldResult(value="SEUMUR HIDUP", confidence=0.85, source="ocr")
    m = DATE_RE.search(value)
    if m:
        return FieldResult(value=m.group(0), confidence=0.8, source="ocr")
    if value:
        # Didn't match either expected shape - still surface it, but at
        # lower confidence since it needs a closer look.
        return FieldResult(value=value, confidence=0.4, source="ocr")
    return FieldResult()


def _extract_kewarganegaraan(lines: list[str]) -> FieldResult:
    """Kewarganegaraan (citizenship) on an Indonesian KTP is always the
    fixed code "WNI" (citizen) or "WNA" (foreigner) - never free text.
    Unlike alamat/pekerjaan/etc, where we have to trust whatever
    follows the label, here anything beyond that 3-letter code is
    guaranteed to be noise (in practice: bleed-through from an
    unrelated caption on the same physical line, as seen on this test
    card's "WNI JAKARTA BARAT" misread)."""
    hit = _find_label_line(lines, "kewarganegaraan")
    if hit is None:
        return FieldResult()
    value = _clean_value(hit[1])
    m = re.search(r"\bWN[AI]\b", value, flags=re.IGNORECASE)
    if m:
        return FieldResult(value=m.group(0).upper(), confidence=0.85, source="ocr")
    if value:
        return FieldResult(value=value, confidence=0.4, source="ocr")
    return FieldResult()


_AGAMA_CODES = ["ISLAM", "PROTESTAN", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "BUDHA", "KONGHUCU"]
_AGAMA_RE = re.compile("(" + "|".join(_AGAMA_CODES) + ")", re.IGNORECASE)
_GOLDARAH_RE = re.compile(r"\b(AB|A|B|O)\b")
_RT_RW_RE = re.compile(r"(\d{1,3}\s*/\s*\d{1,3})")


def _extract_rt_rw(lines: list[str]) -> FieldResult:
    """RT/RW is always digits/digits (e.g. "007/008") - never free
    text. Anchors on that expected pattern within a short window after
    the label, rather than trusting the raw tail regardless of how
    garbled it looks.

    Deliberately NO low-confidence fallback for non-matching text
    (unlike agama/kewarganegaraan below, which keep one for a genuinely
    unrecognized-but-plausible value) - an ensemble across several
    preprocessing variants means every additional variant is another
    chance for pure OCR noise on this line, and there's no such thing
    as a legitimate near-miss RT/RW that doesn't contain a digit/digit
    pattern. A confirmed real failure mode: a badly-OCR'd variant's
    noise ("gros" from a garbled "013/006") getting SOME confidence
    score at all was enough for it to beat a different, correctly-null
    variant's result once merged - returning not_found outright when
    the shape check fails is what avoids that, not just lowering the
    score.
    """
    hit = _find_label_line(lines, "rt_rw")
    if hit is None:
        return FieldResult()
    m = _RT_RW_RE.search(hit[1][:20])
    if m:
        normalized = re.sub(r"\s*/\s*", "/", m.group(1))
        return FieldResult(value=normalized, confidence=0.8, source="ocr")
    return FieldResult()


def _extract_golongan_darah(lines: list[str]) -> FieldResult:
    """Golongan Darah (blood type) is one of a small fixed set - A, B,
    AB, O, or "-" when unknown/not recorded on the card - never free
    text. The same closed-vocabulary approach as agama/kewarganegaraan
    below, and needed for the same reason: a single-letter code is
    exactly the kind of thing that produces spurious matches (an
    unrelated stray letter in noisy OCR text, or the fuzzy label
    fallback landing on the wrong nearby line) if the raw tail were
    trusted directly instead. Only a short window right after the
    label is searched - a blood type is always the very next token,
    so searching further into the line risks matching an unrelated
    single letter elsewhere in it.

    Deliberately NO low-confidence fallback for a non-matching window
    (unlike agama/kewarganegaraan, which keep one) - there's no such
    thing as a legitimate near-miss blood type reading that isn't one
    of exactly five characters, so anything else is noise, not a
    plausible-but-unrecognized value. Returning not_found outright
    (rather than SOME confidence, however low) matters specifically
    because these results get merged across several preprocessing
    variants: any confidence above zero is still enough to beat a
    different, correctly-null variant's result once merged, so the
    fix has to be "never score noise" rather than "score it lower."
    """
    hit = _find_label_line(lines, "golongan_darah")
    if hit is None:
        return FieldResult()
    window = hit[1][:6].upper()
    m = _GOLDARAH_RE.search(window)
    if m:
        return FieldResult(value=m.group(1), confidence=0.75, source="ocr")
    return FieldResult()


def _extract_agama(lines: list[str]) -> FieldResult:
    """Agama (religion) is one of a fixed, known set of values on an
    Indonesian KTP - never free text. Like kewarganegaraan's WNI/WNA
    lookup, anchor on the known code rather than trusting the raw tail
    after the label: a misread ':' on this card's font renders as a
    stray digit fused directly onto the value with no space
    ("Agama 2ISLAM ie" for "Agama : ISLAM"), which a plain
    strip-leading-separators pass can't clean up since a digit isn't a
    separator character - but the known word "ISLAM" is still findable
    as a substring regardless of what's glued onto either side of it.
    """
    hit = _find_label_line(lines, "agama")
    if hit is None:
        return FieldResult()
    m = _AGAMA_RE.search(hit[1])
    if m:
        value = m.group(1).upper()
        if value == "BUDHA":
            value = "BUDDHA"
        return FieldResult(value=value, confidence=0.85, source="ocr")
    value = _clean_value(hit[1])
    if value:
        # Didn't match any known code - still surface it, but flag the
        # lower confidence since it needs a closer look (unrecognized
        # value, or OCR mangled it past recognition).
        return FieldResult(value=value, confidence=0.35, source="ocr")
    return FieldResult()


def parse_ktp(raw_text: str) -> KTPExtractionResult:
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]

    tempat_lahir, tanggal_lahir = _extract_tempat_tanggal_lahir(lines)

    return KTPExtractionResult(
        nik=_extract_nik(lines),
        nama=_extract_simple(lines, "nama", confidence=0.8),
        tempat_lahir=tempat_lahir,
        tanggal_lahir=tanggal_lahir,
        jenis_kelamin=_extract_simple(lines, "jenis_kelamin", cut_before=r"gol\.?\s*darah"),
        golongan_darah=_extract_golongan_darah(lines),
        alamat=_extract_simple(lines, "alamat", confidence=0.6),
        rt_rw=_extract_rt_rw(lines),
        kel_desa=_extract_simple(lines, "kel_desa", confidence=0.75),
        kecamatan=_extract_simple(lines, "kecamatan", confidence=0.75),
        agama=_extract_agama(lines),
        status_perkawinan=_extract_simple(lines, "status_perkawinan"),
        pekerjaan=_extract_simple(lines, "pekerjaan"),
        kewarganegaraan=_extract_kewarganegaraan(lines),
        berlaku_hingga=_extract_berlaku_hingga(lines),
    )


def _cleanliness_score(value: str) -> float:
    """Rough proxy for how trustworthy an OCR string looks - used ONLY
    to break ties between two candidates already at equal confidence
    (see merge_ktp_results). Not a confidence measure on its own: a
    higher fraction of letters/digits/common punctuation (vs. stray
    OCR-noise symbols like '@', '"', '|') suggests a cleaner read,
    nothing more.
    """
    if not value:
        return -1.0
    good = sum(1 for c in value if c.isalnum() or c in " /.-")
    return good / len(value)


def _better_field(a: FieldResult, b: FieldResult) -> FieldResult:
    if a.value is None:
        return b
    if b.value is None:
        return a
    if b.confidence != a.confidence:
        return b if b.confidence > a.confidence else a
    return b if _cleanliness_score(b.value) > _cleanliness_score(a.value) else a


def merge_ktp_results(results: list[KTPExtractionResult]) -> KTPExtractionResult:
    """Combines several parse_ktp() outputs - each from a DIFFERENTLY
    preprocessed version of the same photo (see
    preprocessing.preprocess_for_ktp_ocr) - into one result, picking
    the best candidate independently per field rather than picking one
    "best" attempt wholesale.

    This is worth doing because, on a real degraded photo, different
    preprocessing choices (denoise strength, sharpening, upscale
    factor) each recover a DIFFERENT subset of fields cleanly, with
    little overlap - verified against an actual "holding the card up"
    press photo, where no single variant recovered more than about
    half the card, but the half each one got right barely overlapped
    with the others. Per field: a found value beats a not_found one;
    among found values, higher confidence wins; equal-confidence ties
    are broken by _cleanliness_score, since this parser can return the
    same confidence for two candidates that still differ meaningfully
    in how legible they actually are.
    """
    if not results:
        return KTPExtractionResult()
    if len(results) == 1:
        return results[0]

    merged: dict[str, FieldResult] = {}
    for field_name in KTPExtractionResult.model_fields:
        best = getattr(results[0], field_name)
        for result in results[1:]:
            best = _better_field(best, getattr(result, field_name))
        merged[field_name] = best
    return KTPExtractionResult(**merged)


def reconcile_nik(text_result: FieldResult, region_candidates: list[str]) -> FieldResult:
    """Combines the merged whole-text tiered NIK extraction (_extract_nik,
    run per-variant against the full-page OCR text and merged across
    variants by merge_ktp_results) with zero or more independent,
    digits-only region reads - one per preprocessing variant where
    field_ocr.locate_nik_value_crop found a separate "NIK" label token to
    crop the value from and ocr_engine.image_to_digits then read that crop
    with a digit-only character whitelist, so it can't misread a digit as
    a letter in the first place.

    text_result.value is always either None or already exactly 16 digits
    (see _extract_nik) - region_candidates are NOT pre-filtered by length,
    since field_ocr's crop can occasionally include a stray neighboring
    digit or miss a faint one.

    Two independent methods agreeing is the strongest signal this service
    can produce for a field this important; the two disagreeing is
    itself informative (per the spec this was built against: "if
    candidates disagree substantially, lower confidence rather than
    inventing a value") - handled by degree below rather than as a single
    match/no-match check, since a one-digit slip between two otherwise-
    identical reads is a much weaker signal of trouble than two
    completely different 16-digit numbers.
    """
    valid_region = [c for c in region_candidates if len(c) == NIK_DIGITS]
    text_value = text_result.value

    if text_value and valid_region:
        if text_value in valid_region:
            # Independent whole-text and digit-only-region reads agree -
            # the strongest possible signal for this field.
            return FieldResult(value=text_value, confidence=0.97, source="ocr")
        best_region = min(valid_region, key=lambda c: _levenshtein(c, text_value))
        distance = _levenshtein(best_region, text_value)
        if distance <= 2:
            # Close but not identical (e.g. one digit differs) - keep the
            # whole-text tiered value, since _extract_nik's own tiering
            # already ranks it as the more trustworthy read, but at
            # reduced confidence since the independent region pass didn't
            # fully confirm it.
            return FieldResult(
                value=text_value, confidence=max(0.4, text_result.confidence - 0.25), source="ocr"
            )
        # Substantially different 16-digit numbers from two independent
        # methods - neither can be trusted over the other on looks alone.
        # Per the "don't invent a value" requirement this was built
        # against, surface this as not-found rather than arbitrarily
        # picking one.
        return FieldResult()

    if text_value:
        return text_result

    if valid_region:
        # Whole-text extraction found nothing usable at all, but an
        # independent digit-only region read recovered a clean 16-digit
        # value. Take whichever value the region passes agreed on most
        # (ties broken by list order), always at a confidence capped below
        # what a text+region agreement would earn, since this path has no
        # corroboration from the whole-text side.
        counts: dict[str, int] = {}
        for candidate in valid_region:
            counts[candidate] = counts.get(candidate, 0) + 1
        best_value = max(counts, key=lambda v: counts[v])
        return FieldResult(value=best_value, confidence=0.55, source="ocr")

    return FieldResult()
