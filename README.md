# CompliFi OCR Service

A standalone FastAPI microservice for two things CompliFi's stamp form
currently makes users type by hand:

- **KTP extraction** (`POST /v1/extract/ktp`) — NIK, name, birthplace/date,
  address, etc. from a photo/scan of an Indonesian ID card.
- **Document field extraction** (`POST /v1/extract/document`) — candidate
  document dates and monetary values from a contract/invoice PDF, for the
  `docdate` / `docvalue` fields on the stamp form.
- **Faktur Pajak extraction** (`POST /v1/extract/faktur-pajak`) — header
  (FK) and line-item (OF) fields from an Indonesian Tax Invoice, field-named
  to match CompliFi's e-Faktur bulk-import CSV columns.

This service is intentionally **decoupled from CompliFi**: it doesn't
know about CompliFi's database, sessions, Kong gateway, tenants, or
Peruri integration. It's a pure "send a file, get structured JSON back"
API, guarded by a single shared API key. CompliFi's Express backend calls
it over HTTP and decides what to do with the result.

Engine: **Tesseract** via `pytesseract`, chosen for easy self-hosting (a
single apt package, no GPU, no external API dependency/cost). PDF pages
are rendered with PyMuPDF (pure Python, no poppler binary needed).

## Why a separate service, not a library inside CompliFi

- Tesseract + OpenCV + PyMuPDF is a meaningfully different, heavier
  dependency stack than the existing Node/Express backend — keeping it
  out of that codebase avoids bloating its deploy and avoids a
  Python-in-a-Node-repo situation.
- It can be scaled/restarted independently of the main app (OCR is CPU-
  bound; a burst of scans shouldn't compete with request-serving).
- It can be reused later by other document flows (faktur ingestion,
  batch stamping) without re-plumbing.

## Project layout

```
app/
  main.py               FastAPI app + the /v1/extract and /v1/locate-signature endpoints
  config.py              env-driven settings (API key, languages, limits)
  schemas.py              Pydantic request/response models
  core/                 infra shared by every document type
    ocr_engine.py          thin pytesseract wrapper (swap engines here later)
    preprocessing.py        PDF/image loading + card detect/crop + grayscale/deskew/threshold
    pdf_text.py              native PDF text/table-rule extraction via PyMuPDF
  ktp/                  KTP (Indonesian ID card) extraction
    parser.py              label-based field parser + multi-variant merge + NIK reconciliation
    field_ocr.py             locates a field's value region (currently just NIK) for a
                               second, independent, digits-only OCR pass over just that crop
    image_quality.py          blur/resolution/lighting/glare assessment of the source photo
    pipeline.py               wires the above together end-to-end - the actual logic
                               behind /v1/extract/ktp, kept separate from main.py so it's
                               testable without FastAPI
  document/             generic documents + signature placement
    parser.py               date/currency candidate extraction for docs
    signature_locator.py      signature-placement geometry lookup
  faktur_pajak/          Faktur Pajak (Indonesian tax invoice)
    parser.py               FK/OF field parser
tests/
  test_extraction.py    end-to-end tests against samples/
samples/                 synthetic + real test images (see "Testing" below)
scripts/
  run.sh                one-command local runner (creates a venv, installs deps, starts uvicorn)
deploy/                  production deployment templates - pick ONE:
  pm2/ecosystem.config.js   for a host already running CompliFi's other services under PM2
  systemd/ocr-service.service  for a host that isn't
Dockerfile               container image (tesseract + deps baked in)
docker-compose.yml        one-command containerized run: `docker compose up`
Makefile                  `make help` for a menu of the above
```

Each subpackage is grouped by WHAT it's about, not by what kind of file
it is - `ktp/` holds parsing, region-OCR, quality-assessment, and
pipeline-wiring code together because those four files only ever change
together when the KTP pipeline changes, not because they share a
technical role. `core/` is the one exception: it holds the low-level,
document-type-agnostic infrastructure (the actual OCR engine call,
image preprocessing, PDF text extraction) that `ktp/`, `document/`, and
`faktur_pajak/` all build on top of.

## Local setup

Quickest path - one command handles the venv, dependencies, and `.env`:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-ind   # system dependency, not pip-installable
./scripts/run.sh          # or: make run
```

`scripts/run.sh` creates `.venv/` and installs `requirements.txt` into it
the first time, copies `.env.example` to `.env` if you don't have one
yet (fine for local dev - fill in `OCR_SERVICE_API_KEY` before exposing
this beyond localhost), then starts uvicorn with auto-reload. Re-running
it later just starts the server - it doesn't recreate anything that's
already there. `./scripts/run.sh prod` runs without reload, matching how
the Dockerfile/PM2/systemd all start it (see "Deployment" below).

Equivalent manual steps, if you'd rather not run a script sight-unseen
or need to poke at one step in isolation:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-ind

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in OCR_SERVICE_API_KEY for anything beyond local dev

uvicorn app.main:app --reload --port 8088
```

Check it's alive:

```bash
curl http://127.0.0.1:8088/health
# {"status":"ok","tesseract_version":"5.3.4","languages":["eng","ind","osd"]}
```

Interactive API docs (Swagger UI) are auto-served at `/docs`.

## Deployment

Three ways to run this in production - pick whichever matches how the
rest of your infrastructure is already run; none of them need the others.

**Docker Compose** (bundles tesseract into the image - nothing to
install on the host besides Docker itself):

```bash
cp .env.example .env   # fill in OCR_SERVICE_API_KEY first
docker compose up --build -d
```

`restart: unless-stopped` in `docker-compose.yml` plus the Dockerfile's
`HEALTHCHECK` mean Docker restarts the container on crash on its own; add
Docker itself to your host's boot sequence (`systemctl enable docker`,
already the default on most distros) to also survive a reboot. `make
docker-build` / `make docker-run` do the same thing without Compose, if
you need the image standalone (e.g. to push it somewhere).

**PM2** (`deploy/pm2/ecosystem.config.js`) - for a host that already runs
CompliFi's other Node/Express processes under PM2 and should manage this
one the same way:

```bash
make setup   # creates .venv/ and installs deps, without starting anything
pm2 start deploy/pm2/ecosystem.config.js
pm2 save   # persist across a `pm2 resurrect` / reboot
```

**systemd** (`deploy/systemd/ocr-service.service`) - for a host running
neither of the above:

```bash
make setup   # or the manual venv steps above
sudo cp deploy/systemd/ocr-service.service /etc/systemd/system/
# edit WorkingDirectory/ExecStart/User in the copied file first - the
# checked-in one has placeholder paths, see its own comments
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-service
```

All three run the exact same `uvicorn app.main:app --host 0.0.0.0 --port
8088` underneath - the choice is purely about which process manager fits
the rest of your deployment, not a functional difference in the service
itself.

Whichever you pick: put it behind the same Nginx/Kong layer as the rest
of CompliFi if you want a single ingress point, or keep it on an
internal-only port reached solely from the Express backend (recommended
- no reason for this service to be internet-facing given it never runs
unauthenticated; see "Security notes" below).

## API

Both extraction endpoints take a `multipart/form-data` upload named
`file` (PDF or JPEG/PNG/WebP), and an optional `include_raw_text=true`
query param for debugging.

### `POST /v1/extract/ktp`

```bash
curl -X POST http://127.0.0.1:8088/v1/extract/ktp \
  -H "X-API-Key: $OCR_SERVICE_API_KEY" \
  -F "file=@ktp_photo.jpg;type=image/jpeg"
```

Returns `ktp.nik`, `ktp.nama`, etc., each as `{ value, confidence, source }`.
`confidence` is a 0–1 heuristic (see `ktp/parser.py`), not a calibrated
probability — treat anything below ~0.6 as "flag for review."

Fields returned: `nik`, `nama`, `tempat_lahir`, `tanggal_lahir`,
`jenis_kelamin`, `golongan_darah` (blood type), `alamat`, `rt_rw`,
`kel_desa` (Kelurahan/Desa), `kecamatan`, `agama`, `status_perkawinan`,
`pekerjaan`, `kewarganegaraan`, `berlaku_hingga`.

`agama` and `kewarganegaraan` are matched against a **closed
vocabulary** (`ISLAM`/`PROTESTAN`/`KRISTEN`/`KATOLIK`/`HINDU`/
`BUDDHA`/`KONGHUCU`; `WNI`/`WNA`) rather than just taking whatever text
trails the label — both are fixed-set fields on a real KTP, so
anchoring on the known value is what lets the parser recover the
correct one even when a misread colon glues stray characters directly
onto it (e.g. Tesseract turning `Agama : ISLAM` into `Agama 2ISLAM ie`
on a real test photo — no space at all between the noise and the real
value, so a plain "strip separator characters" pass can't clean that
up, but the substring `ISLAM` is still findable regardless of what's
stuck to either side of it). Every other free-text field (`alamat`,
`pekerjaan`, `kel_desa`, etc.) still just takes the cleaned label tail,
since there's no fixed vocabulary to anchor those on.

**This endpoint detects and crops the card itself before OCR** —
`preprocess_for_ocr(..., detect_card=True)`, only ever enabled here,
not on `/v1/extract/document`. A flat scan/crop of just the KTP (most
of this service's original testing) doesn't need this, but a real
submission is very often a photo of someone *holding up* their card —
verified against an actual press photo where the card was maybe a
third of the frame, tilted, next to the person's own face. Run through
the OCR pipeline unmodified, that photo came back with **every single
field null** — not because the text-matching regexes were wrong, but
because Tesseract (and this module's own deskew step, whose rotation
estimate is derived from every non-background pixel in the frame) was
working off the *entire* photo, face and background included, with the
actual card text a small, tilted fraction of it. `_detect_and_crop_card`
(in `core/preprocessing.py`) finds the largest card-shaped quadrilateral via
`cv2` contour detection, perspective-warps it to a flat top-down crop,
and `_upscale_if_small` enlarges it if still small — before any of the
existing grayscale/deskew/denoise steps run. It falls back to the
original, uncropped image whenever no sufficiently large, sufficiently
card-shaped (~1.585 long/short ratio, a KTP's real aspect ratio)
quadrilateral is found, so a flat scan (which won't have a smaller
card-shaped region *within* it) passes through unaffected.

Even with a good crop, a sufficiently blurry source photo can still
garble a **label itself**, not just the value after it — an inserted
space splitting `Agama` into `Aga ma`, a misread letter turning
`Perkawinan` into `Pemawnan`. `ktp/parser.py`'s `_find_label_line` only
falls back to **edit-distance label matching** (`_find_label_line_fuzzy`)
when the exact regex found that label nowhere in the whole document —
never instead of the exact match, and never on a line some other field
already exactly claimed (otherwise a correctly-OCR'd `Alamat` line, an
easy near-miss for the 5-letter `Agama` target, could get mis-claimed
by agama's fuzzy search purely because it's edit-distance 1 away).

Realistically: this doesn't turn a blurry photo into a clean scan.
Digit-heavy fields (NIK, exact birth date) are still frequently
unrecoverable from real motion/focus blur — no preprocessing recovers
information that genuinely isn't legible in the source. What it does
fix is the difference between *every* field coming back null and the
fixed-vocabulary and clearly-labelled fields (`agama`,
`kewarganegaraan`, `status_perkawinan`, etc.) still coming through
correctly despite the photo being far from ideal.

**This endpoint also runs the whole pipeline SEVERAL times per image,
with different preprocessing, and merges the results.** A single
denoise-strength/sharpening/upscale-factor choice is not one-size-
fits-all on a real degraded photo: testing against the same held-up
KTP photo showed different variants recovering different, barely-
overlapping subsets of fields — one variant alone recovered
`status_perkawinan` but not `kecamatan`, another got `kecamatan` but
not `status_perkawinan` — with no single variant recovering more than
about half the card. `preprocessing.preprocess_for_ktp_ocr` returns
seven differently-processed versions of the same card crop (five
denoise/sharpen/upscale variants plus two Wiener-deconvolution
variants — see below); the endpoint runs OCR + `parse_ktp` on each,
and `ktp/parser.merge_ktp_results` combines them field-by-field: a
found value beats a `not_found` one, higher confidence wins between
two found values, and a cleanliness heuristic (fewer stray OCR-noise
symbols) breaks ties at equal confidence — e.g. preferring a clean
`"BARUGA"` over a `"@ARUGA"` both extracted at the same confidence.
This costs roughly 7x the OCR latency of a single pass (several
seconds becomes closer to fifteen-plus) — acceptable for a one-off
identity-document upload, not something to reuse elsewhere.
`preprocess_for_ktp_ocr` already skips the ensemble (one pass only)
when card detection doesn't find anything to crop, since a flat scan
doesn't have the "which preprocessing choice reads a blurry photo
best" problem to begin with.

**Two of the seven variants are classical (non-ML) deblurring, via
Wiener deconvolution** (`_wiener_deconvolve` in `core/preprocessing.py`,
implemented directly with `cv2`/`numpy` FFTs — no new dependency) —
one assuming a defocus (out-of-focus lens) blur shape, one assuming
linear motion blur. Unlike denoise/sharpen, which just accentuate
existing edges, deconvolution attempts to mathematically reverse a
*specific, assumed* blur kernel. Tested honestly rather than just
bolted on: against the one real held-up-photo sample, neither kernel
recovered any field that every other variant also missed — this
photo's degradation looks more like genuine out-of-focus blur plus
JPEG compression artifacting than a kernel either assumed shape
matches closely, and Wiener deconvolution is weakest exactly there (a
wrong kernel guess mostly adds ringing, not clarity). They're kept in
the ensemble anyway, at one kernel each, as cheap additional diversity
for a *different* photo whose blur happens to be closer to one of
these two shapes — bounded extra cost (two more OCR passes), and
`merge_ktp_results`'s per-field confidence rules mean a deconvolution
variant that helps nothing can only ever be ignored, never chosen over
a better result from a different variant.

That safety net had to be earned, though, not assumed: an early
version of this ensemble let a deconvolution variant's garbled `rt_rw`
text ("gros", from a genuine "013/006") get *some* confidence from a
low-confidence fallback path, and that alone was enough to beat a
different, correctly-null variant's result once merged — silently
turning a correct "not found" into confident-looking wrong data. The
fix wasn't a smarter merge rule; it was removing that fallback
entirely from `rt_rw`/`golongan_darah` (`_extract_rt_rw`/
`_extract_golongan_darah`), so non-matching text on those two fields
returns `not_found` outright rather than *some* score, however low —
see `test_ktp_deconvolution_variants_never_introduce_wrong_data` for
the regression test this produced. `agama`/`kewarganegaraan` keep
their own low-confidence fallback, deliberately: an unrecognized-but-
plausible religion/citizenship value is a real (if rare) possibility
those two anchor against a closed vocabulary for, unlike a blood type
or an RT/RW number, which have no legitimate "near miss" shape at all.

The response's `warnings` explicitly lists which fields, if any,
weren't recognized by ANY of the seven variants (`"Not recognized even
after multiple preprocessing attempts: ..."`) and which populated
fields came back at low confidence — rather than leaving the caller to
infer that from a wall of `null`s and numbers.

**NIK gets a second, independent, digits-only OCR pass, cross-checked
against the whole-text read.** Everywhere else in this service, OCR
runs once per preprocessing variant over the WHOLE page, and
`ktp/parser.py` finds each field's value by pattern-matching the
resulting text. For NIK specifically — a fixed-format, security-
sensitive 16-digit number — `field_ocr.locate_nik_value_crop` also
finds the pixel region immediately to the right of the "NIK" label
(from Tesseract's own word-level bounding boxes,
`ocr_engine.image_to_data`) and re-runs OCR on JUST that crop with a
digits-only character whitelist (`ocr_engine.image_to_digits`). A
whitelisted pass can't output a letter at all, so a glyph that's
ambiguous between e.g. "0"/"O" or "1"/"I" is forced to its digit
reading directly, rather than needing to be recognized as a letter
first and then corrected (or, as the whole-text path used to do,
losing it entirely — see below). Each crop is actually read TWICE —
once at Tesseract psm 7 ("single line"), once at psm 8 ("single
word") — since a NIK value crop is always both of those at once, and
in practice the two occasionally disagree with each other on a
marginal crop (found via a real photo: psm 7 misread the crop's last
digit while psm 8, on the very same crop, read it correctly). Both
readings are added to the candidate pool rather than picking one psm
mode as "the" region read — `ktp/parser.reconcile_nik` then
combines the merged whole-text candidate with however many region
candidates the ensemble's variants (and both psm passes) produced:

- Whole-text and region agree → confidence `0.97`, the strongest
  signal this service can produce for this field.
- They differ by only a digit or two (edit distance ≤ 2) → keep the
  whole-text value (already the more-trusted read per its own internal
  tiering) but at meaningfully reduced confidence, since the
  independent pass didn't fully confirm it.
- They differ substantially (two genuinely different 16-digit numbers)
  → `null`, not a guess. Per the extraction spec this was built
  against — "do not fabricate OCR values… return null rather than
  inventing a number" — two independent methods actively disagreeing
  is not a case to arbitrate by picking whichever "looks" right.
- Only the region pass found a valid 16-digit value → surfaced at a
  capped confidence (`0.55`), since it has no whole-text corroboration.
- Neither found a clean 16-digit value anywhere → `null`.

This directly fixes a real bug: the whole-text tiered extractor used
to have a last-resort tier that accepted an off-by-one-length digit
run (15 or 17 digits) at low confidence rather than nothing — which
is itself a fabricated value (it fails NIK's own strict format rule),
just wearing a low confidence score. Against this service's real
held-up-photo sample, that tier used to return a 15-digit non-NIK
string; it now correctly returns `null` (see
`test_ktp_pipeline_nik_never_returns_an_invalid_length_value` and the
`ktp/parser._extract_nik` Tier 4 comment). The same whole-text tier
also now corrects common single-glyph digit/letter confusions
("O"→"0", "I"/"l"→"1", "S"→"5", "B"→"8") BEFORE stripping non-digit
characters, rather than just deleting the letter and silently
shortening the digit count — a second, smaller bug the same real
photo's failure mode surfaced.

**`_clean_value`'s stray-punctuation stripping now covers apostrophes,
quotes, and a few other common OCR-noise characters, not just
`":.-_"`.** Found via the same flat-template photo: a fuzzy-matched
label's tail can start with a leftover misread character before the
real value (e.g. `"'— : PEGADUNGAN"`, from "Kel/Desa" being read as
"KellDesa" with no real "/", so the exact-match regex missed it and
the fuzzy fallback's cut point landed right after a stray apostrophe).
`_clean_value` strips a leading/trailing RUN of separator-ish
characters in one pass rather than `.strip()`'s fixed set (which stops
dead at the first character outside it) — but stopping dead is exactly
what happened here anyway, because the old character class didn't
include `'` at all, so the strip matched zero characters at the string
start and left the whole stray prefix attached. See
`test_clean_value_strips_apostrophe_prefixed_stray_punctuation`.

**A lightweight image-quality assessment runs alongside OCR and is
returned as `quality`.** `image_quality.assess_quality` computes, on
the actual pixels handed to Tesseract (post-crop, post-upscale):
`blur_score` (0–1, from the variance of the Laplacian — NOT an OCR
confidence proxy, just sharpness), `resolution_ok`, `lighting_ok`,
`glare_detected` (a spatially concentrated near-saturated blob, not
just "a lot of bright pixels" — a plain white card background
legitimately has plenty of those without any real reflection on it;
an earlier version of this check false-positived on exactly that, see
`test_assess_quality_does_not_flag_a_plain_white_background_as_glare`),
`document_detected`/`perspective_corrected` (whether card-detect/crop
found anything), and `document_area_ratio` (cropped card area ÷
original frame area, when detected). These feed directly into
`warnings` (e.g. `"KTP detected but the source image appears
blurry."`) so a thin, null-heavy result can be told apart from a
parser bug — and so a caller can decide whether to ask for a re-scan
before even looking at individual field confidences. Like every other
threshold in this pipeline (see "Known limitations" below), the exact
cutoffs are a starting point chosen against this service's bundled
real photo sample, not a calibrated, universal standard.


### `POST /v1/extract/document`

```bash
curl -X POST http://127.0.0.1:8088/v1/extract/document \
  -H "X-API-Key: $OCR_SERVICE_API_KEY" \
  -F "file=@contract.pdf;type=application/pdf"
```

Returns a top-pick `document.docdate` / `document.docvalue` plus the
full `docdate_candidates` / `docvalue_candidates` lists, since a contract
usually contains more than one date or amount — let the user pick if
there's more than one candidate rather than trusting the top pick
blindly.

### `POST /v1/extract/faktur-pajak`

```bash
curl -X POST http://127.0.0.1:8088/v1/extract/faktur-pajak \
  -H "X-API-Key: $OCR_SERVICE_API_KEY" \
  -F "file=@faktur_pajak.pdf;type=application/pdf"
```

Returns one header record (buyer/seller identity, invoice number,
period, DPP/PPN/PPnBM totals, down-payment fields, etc.) plus a
`line_items` array — one entry per row of the invoice's item table.
Field names are `lower_snake_case` here (schemas.py convention); they
map 1:1 to the `UPPER_SNAKE_CASE` columns CompliFi's bulk-import CSV
uses:

| Response field | CSV column | Response field | CSV column |
|---|---|---|---|
| `npwp_wp` | `NPWP_WP` | `fg_uang_muka` | `FG_UANG_MUKA` |
| `id_tku_wp` | `ID_TKU_WP` | `nomor_faktur_um_sebelumnya` | `NOMOR_FAKTUR_UM_SEBELUMNYA` |
| `kd_jenis_transaksi` | `KD_JENIS_TRANSAKSI` | `uang_muka_dpp` | `UANG_MUKA_DPP` |
| `fg_pengganti` | `FG_PENGGANTI` | `uang_muka_dpp_lain` | `UANG_MUKA_DPP_LAIN` |
| `nomor_faktur` | `NOMOR_FAKTUR` | `uang_muka_ppn` | `UANG_MUKA_PPN` |
| `masa_pajak` / `tahun_pajak` | `MASA_PAJAK` / `TAHUN_PAJAK` | `uang_muka_ppnbm` | `UANG_MUKA_PPNBM` |
| `tanggal_faktur` (ISO `YYYY-MM-DD`) | `TANGGAL_FAKTUR` | `referensi` | `REFERENSI` |
| `npwp` / `jenis_identitas` / `nik_nomor_passport` / `kode_negara` | buyer fields of the same name | `kode_dokumen_pendukung` | `KODE_DOKUMEN_PENDUKUNG` |
| `nama` / `email_pembeli` / `alamat_pembeli` / `tku_pembeli` | buyer fields of the same name | `tax_code` | `TAX_CODE` |
| `jumlah_dpp` / `jumlah_dpp_lain` / `jumlah_ppn` / `jumlah_ppnbm` | fields of the same name | `faktur_pajak_dummy` | `FAKTUR_PAJAK_DUMMY` |

Each `line_items[]` entry maps the same way to an `OF` row (`kode_objek`
→ `KODE_OBJEK`, `harga_satuan` → `HARGA_SATUAN`, etc.) — see
`schemas.py::FakturPajakLineItem` for the full list.

**Two real layouts are supported side by side** (see
`faktur_pajak/parser.py`'s module docstring for the full breakdown): a real
DJP Coretax-issued Faktur Pajak — verified against an actual production
PDF — and a simpler layout used for early testing. They differ more
than you'd expect: colon-separated labels vs. not, full-word labels
("Dasar Pengenaan Pajak") vs. abbreviations ("DPP"), an invoice date
that only appears as an Indonesian-month-name date next to the
e-signature (no "Tanggal Faktur" label at all) vs. an explicit
labelled date, a NITKU embedded in the address as `#<22 digits>` with
no label of its own vs. an explicit `NITKU` label, one-row-per-item
tables vs. multi-line item blocks with a per-line PPnBM breakdown, and
blank fields printed as a literal `-` vs. omitted entirely. A handful
of fields — `jenis_identitas` and `fg_uang_muka` — are **derived**
rather than read off an explicit label when the Coretax layout doesn't
print one directly (documented inline in `faktur_pajak/parser.py`): which of
NIK/Nomor Paspor/NPWP/Identitas Lain is actually non-blank determines
`jenis_identitas`, and whether "Dikurangi Uang Muka yang telah
diterima" has an amount next to it determines `fg_uang_muka`. Both are
structural inferences from what's already on the page, not invented
data.

**`validation` recognizes DJP's "DPP Nilai Lain" scheme.** Some
transaction categories (e.g. certain telecom/voucher products) compute
DPP as `(Harga Jual − Potongan) × 11/12`, giving an effective 11% PPN
rate once the standard 12% applies on top — a real invoice validated
against this service confirmed it. The validator accepts either that
formula or the plain `DPP = net-of-discount` convention before
flagging a mismatch, and similarly accepts a line's `HARGA_TOTAL`
being either gross or net-of-discount (some documents apply the
discount only at the DPP level, not per line) — so `WARNING` means an
actual numeric inconsistency, not just "used the other document's
convention."

**Per-line tax fields are usually `null`, but not always.** Most
Faktur Pajak print DPP/DPP Nilai Lain/PPN/PPnBM only at the invoice
level, not per line item — in that case, rather than splitting those
totals pro-rata across `line_items` (no confirmed business rule for
that yet), this service leaves `check_dpp_lain`/`dpp`/`dpp_lain`/
`tarif_ppn`/`ppn`/`tarif_ppnbm`/`ppnbm` as `null`. The real Coretax
layout, however, DOES print a per-line PPnBM rate/amount inline in the
item description ("PPnBM (0,00%) = Rp 0,00") — `tarif_ppnbm`/`ppnbm`
are populated from that when present, since that's a genuine per-line
figure, not an allocation. `validation.missing_fields` and the
response's top-level `warnings` call out whichever fields are missing
for this reason, so it isn't mistaken for a parsing failure.

**`validation.calculation_status`** is one of `VALID` / `WARNING` /
`INVALID` / `NOT_CHECKED`, with human-readable `calculation_notes`
explaining each check performed. This is a heuristic sanity check, not
a substitute for an accountant's review — treat a `WARNING` as "worth
a second look," not "reject the invoice."

**Prefers the PDF's own embedded text over OCR.** A Faktur Pajak —
whether issued via DJP Coretax or generated by CompliFi itself — is
almost always a digitally-generated PDF, so this endpoint reads
PyMuPDF's native text extraction directly (same `has_extractable_text`
check `/v1/locate-signature` already uses) instead of rasterizing the
page and running Tesseract. It only falls back to OCR for a genuinely
scanned/photographed invoice or a plain image upload — in that case
the response's `warnings` says so, since OCR misreads (digit/letter
confusion, broken decimal separators) are far more likely on
identifiers like `NOMOR_FAKTUR`/`NPWP` than on native PDF text.

### `POST /v1/locate-signature`

Finds where to stamp a signer's signature specimen: the blank cell
directly above their printed name in a signature block (e.g. the
"Pemohon" column on a Peruri-style form). Takes `file` plus a
`signer_name` form field (their full legal name — matching tolerates
the document abbreviating it, e.g. "Muhammad Gifar Z." still matches
"Muhammad Gifar Zaini").

```bash
curl -X POST http://127.0.0.1:8088/v1/locate-signature \
  -H "X-API-Key: $OCR_SERVICE_API_KEY" \
  -F "file=@pernyataan_kerahasiaan.pdf;type=application/pdf" \
  -F "signer_name=Muhammad Gifar Zaini"
```

```json
{
  "signer_name": "Muhammad Gifar Zaini",
  "matches": [
    {
      "page_index": 0,
      "matched_text": "Muhammad Gifar Z.",
      "match_confidence": 0.97,
      "x0": 134.8, "y0": 667.7, "x1": 222.7, "y1": 713.1,
      "page_width": 595.3, "page_height": 841.9,
      "method": "pdf_native"
    }
  ],
  "warnings": []
}
```

**How it works — this is a geometry lookup, not OCR, for a normal
CompliFi document.** A digitally generated PDF (the Peruri template,
any CompliFi-issued form) already stores every printed name and every
table rule as an exact-coordinate object — no OCR needed. `core/pdf_text.py`
reads the name text and the table's rule lines directly;
`document/signature_locator.py` finds the printed name matching `signer_name`
and computes the blank cell bounded by the rules around it (the cell
directly above the name, between the row separator above it and the
column's left/right borders).

Two things worth knowing before wiring this up:

- **The signer's name can appear more than once** (e.g. once as an
  identity field like "Nama: ..." near the top, once — possibly
  abbreviated — under the actual signature line). Text-similarity alone
  favors the unabbreviated header hit, which is the *wrong* one for
  placement, so the locator prefers whichever match sits inside a fully
  ruled table cell over one that merely scores higher on text
  similarity. See the regression test in `tests/test_extraction.py`
  (`test_signature_locator_prefers_signature_block_over_header_field`)
  for the exact case this fixes.
- **`method` on each match tells you which coordinate space you're in.**
  `"pdf_native"` results are PDF points with a top-left origin — use
  them directly if you're stamping with a PDF library (e.g. `pdf-lib`,
  noting most PDF libraries use a *bottom-left* origin, so flip
  `y` to `page_height - y`). `"ocr_scanned"` results are pixel
  coordinates at whatever DPI the page was rasterized at — this is the
  fallback path for a genuinely scanned/photographed document with no
  extractable text at all, and its confidence is deliberately capped
  lower; treat it as pre-fill for a human to confirm/drag into place,
  never as a final coordinate.
- **A document can have the same signer's name on more than one page**
  (a multi-page contract re-signed per page, or — as seen in a test
  upload — the same page duplicated across the PDF). `matches` returns
  one entry per page where the name was found, not just the first hit.

## Integrating with CompliFi's Express backend

Call this service from the Node backend (not directly from the Next.js
frontend, so the API key never reaches the browser):

```javascript
// e.g. src/domain/ocr/service.js
const FormData = require("form-data");
const fetch = require("node-fetch");

async function extractKtp(fileBuffer, mimeType) {
  const form = new FormData();
  form.append("file", fileBuffer, { contentType: mimeType, filename: "ktp" });

  const res = await fetch(`${process.env.OCR_SERVICE_URL}/v1/extract/ktp`, {
    method: "POST",
    headers: { "X-API-Key": process.env.OCR_SERVICE_API_KEY },
    body: form,
  });
  if (!res.ok) throw new Error(`OCR service returned ${res.status}`);
  return res.json();
}
```

Expose a thin CompliFi endpoint (e.g. `POST /exclusive/api/v1/ocr/ktp`)
that proxies to this, so the frontend's `lib/api.ts` gets a normal
same-origin call and the OCR service URL/key stay server-side only —
same pattern already used for the Kong apikey proxying you did in the
frontend security hardening work.

On the frontend (`app/dashboard/stamp/page.tsx`), add a "Scan Otomatis"
action next to the identity/document fields that calls the new proxy
endpoint and pre-fills `setIdentityNumber`, `setClientName`, `setDocdate`,
`setDocValue` — but keep the fields editable and visibly marked as
OCR-derived (e.g. using each field's `confidence`/`source`) rather than
locking them. The 16-digit client-side validation you already have in
`validateForm()` stays as the final gate either way.

For signature placement, call `/v1/locate-signature` with the signer's
name once you know it (e.g. after the KTP/identity step), then use the
returned box to position the signature-specimen overlay on the stamp
preview canvas — converting PDF points to the canvas's pixel space via
`page_width`/`page_height` in the response, same as you already do for
the e-Meterai stamp's own placement. Still let the user drag/nudge it;
a computed box is a good default, not a guarantee the template layout
never varies.

## Security notes

- **Nothing is persisted.** Uploaded files exist only in memory for the
  duration of the request; nothing is written to disk. If you add
  logging, make sure `OCR_DEBUG_LOG_RAW_TEXT` stays `false` in
  production — a KTP's OCR text contains a live NIK.
- **Auth is a single shared API key** (`X-API-Key` header), meant for
  server-to-server calls from CompliFi's backend only. Don't expose this
  service's key to the browser.
- **Confidence scores are heuristics**, not guarantees — every field this
  service returns should remain user-editable in the CompliFi UI, never
  silently auto-submitted.

## Testing

```bash
pytest tests/ -v
```

The included `samples/` images are synthetic (rendered with a clean
DejaVu font), which is enough to catch pipeline-level regressions — this
is actually how a real bug was caught while building this: a deskew
step that worked fine on dense KTP-card content span rotated a
sparse-text document 90° and destroyed it (see the safety clamp and
comment in `core/preprocessing.py`). Before trusting this for real scans:

1. Collect a batch of real (test/consented) KTP photos and CompliFi
   documents — phone-camera lighting/angle varies far more than a
   rendered PNG.
2. Run them through `/v1/extract/*` with `include_raw_text=true` and
   compare `raw_text` against what Tesseract actually saw before
   blaming the parser for a bad extraction.
3. Tune `LABEL_PATTERNS` in `ktp/parser.py` and the date/currency regexes
   in `document/parser.py` against whatever OCR misreads show up on real
   cards — this is the part most worth iterating on.

## Known limitations (found via real-KTP testing)

- **Card detection can still fail safely-but-uselessly on a photo whose
  true card boundary is too low-contrast for Canny to close into one
  contour** (see `_detect_and_crop_card`'s docstring and
  `_MAX_CORNER_ANGLE_DEVIATION`). This was a real bug, not a hypothetical
  one: against a clean, flat, studio-style KTP template photo
  (`samples/sample_ktp_flat_template.jpg`), the actual card edge never
  closed into a contour at all, so the largest contour Canny DID find was
  an unrelated internal watermark graphic - a heavily sheared
  quadrilateral that happened to also clear the area-share and aspect-
  ratio checks, producing a perspective warp that sheared every label out
  of line with its own value. A rectangularity check (rejecting any
  candidate whose corners deviate too far from 90°) now catches that
  specific failure and falls back to the untouched image rather than
  returning a corrupted crop - but on a photo where the crop WOULD have
  genuinely helped (a real perspective/rotation problem) and this same
  low-contrast-boundary issue happens to occur, the result is "no crop
  happened" rather than "the crop that should have happened, happened" -
  fail-safe, not a full fix. A more thorough fix would involve a
  detection strategy that doesn't depend on the card boundary itself
  producing a strong Canny edge (e.g. background-subtraction via flood-
  fill from the frame corners, or a learned document-detection model) -
  out of scope for this pass; see `test_detect_and_crop_card_rejects_a_
  non_rectangular_distractor` for the regression test built from this
  exact photo.
- **The NIK region cross-check only fires when Tesseract's own tokenizer
  cleanly separates the "NIK" label from its value** (see
  `field_ocr.locate_nik_value_crop`'s docstring). When a misread colon
  glues the label and the first digit(s) into one token — the exact
  failure mode `ktp/parser.py`'s `_find_label_line_fuzzy`/NIK Tier 1-4
  handling was built around — this pass is skipped for that variant
  rather than guessing where the label ends inside the glued token,
  since a wrong guess there would silently feed the digits-only OCR a
  crop that still starts mid-label. The whole-text tiered path still
  runs regardless; only the extra corroboration is unavailable on that
  particular variant/photo.
- **`reconcile_nik`'s "wide disagreement → null" rule can occasionally
  discard a value one of the two methods actually got right.** Two
  independent reads landing on two substantially different 16-digit
  numbers is treated as "neither can be trusted" rather than picking a
  side — the safer default for an identity number, but it does mean a
  photo where (say) the whole-text read is correct and the region crop
  happened to OCR a stray extra/missing digit from crop-boundary noise
  can end up `null` instead of surfacing the correct value at reduced
  confidence. Only verified end-to-end against this service's one real
  held-up-photo sample and one clean flat scan — tune the edit-distance
  cutoff (currently 2) against more real (test) photos before treating
  it as final.
- **`ktp/image_quality.py`'s thresholds (blur/resolution/lighting/glare) are
  a starting point, not a calibrated standard** — chosen against the
  same two bundled real/synthetic samples as everything else in this
  service, the same caveat the multi-variant ensemble's own variant
  list carries. In particular, a synthetic, plain-white-background test
  render (like `samples/sample_ktp.png`) reads as `lighting_ok: false`
  purely from its unusually high mean brightness — a real photographed
  card, which is never actually pure white, shouldn't hit this in
  practice, but it's a real gap between the bundled synthetic sample and
  a real card worth knowing about before trusting the exact cutoff.
  `glare_detected` specifically looks for a spatially CONCENTRATED
  bright blob rather than raw overexposed-pixel fraction, precisely to
  avoid flagging a plain bright/white background as glare (see
  `image_quality._glare_detected` and the regression test
  `test_assess_quality_does_not_flag_a_plain_white_background_as_glare`)
  — but a real photo with genuinely diffuse, all-over overexposure
  (rather than a localized hotspot) would still slip past this
  specific check, and only show up via `lighting_ok`'s brightness/
  contrast thresholds instead.
- **Preprocessing is intentionally light by default.** An earlier version
  forced adaptive thresholding on every image; testing against a real
  KTP photo showed it actively corrupting clean, high-contrast cards —
  it turned the ":" after "NIK" into a digit-like glyph that Tesseract
  read as a literal "2", corrupting the identity number. Grayscale +
  deskew + light denoise (no forced threshold) reads that same card
  correctly, since Tesseract already does its own internal
  binarization. `preprocess_for_ocr(image, force_threshold=True)` is
  still available for the genuinely hard case this was meant for — a
  real phone photo with uneven lighting/shadow — if you hit one where
  the default is unreadable.
- **Multi-column bleed-through.** Some KTP layouts put a caption or
  second field on the same physical line as another field (e.g. "Jenis
  Kelamin" and "Gol. Darah" sharing a line, or a photo caption sitting
  next to "Kewarganegaraan"/"Berlaku Hingga"). The line-based parser
  handles the specific cases seen so far (`jenis_kelamin` cuts before
  "Gol. Darah"; `kewarganegaraan` is constrained to the known WNI/WNA
  codes; `berlaku_hingga` extracts just the date or "SEUMUR HIDUP"
  rather than the raw tail) but a genuinely different layout could
  still bleed two fields together. Watch for this in `agama` and
  `alamat` especially, which stay free-text with no such constraint.
- **Watermarks/background patterns can leave a stray trailing
  character** on an otherwise-correct field (seen: `"ISLAM K"` instead
  of `"ISLAM"`). Low-value to special-case further without more real
  samples showing a pattern — currently left as a minor, low-confidence
  artifact for the user to clean up on review.
- **Card detection assumes one dominant, roughly ID-1-shaped
  (~1.585 long/short ratio) quadrilateral in frame.** Verified against
  one real "holding the card up" photo — a different photo where the
  card is heavily occluded (fingers over a corner), extremely tilted
  (beyond ~15° after the existing MAX_AUTO_DESKEW_DEGREES clamp), or
  where something else in the shot is coincidentally card-shaped and
  larger, could still fail to crop correctly or crop the wrong thing.
  It fails safe either way — falling back to the original, uncropped
  image rather than a bad crop — so the worst case is the same
  "everything null" result as before this fix, not a worse one.
- **Fuzzy label matching (`_find_label_line_fuzzy`) is a last resort,
  not a general OCR-error corrector.** It only fires when a field's
  exact label regex found nothing anywhere in the document, and it
  only fixes the LABEL being garbled — a genuinely blurry photo can
  still garble the VALUE after a perfectly-readable label (this is
  the case for this service's NIK/birth-date digits on the one real
  held-up-photo sample tested against: label found fine, digits
  themselves not reliably recoverable). No preprocessing step
  recovers information that's genuinely illegible in the source photo.
- **The multi-variant ensemble raises the ceiling, it doesn't
  guarantee every field.** On the one real held-up-photo sample tested
  against, all 7 preprocessing variants (including both deconvolution
  ones) still came back null for
  `tempat_lahir`/`tanggal_lahir`/`jenis_kelamin`/`golongan_darah`/
  `rt_rw` — those specific lines were too degraded in the source photo
  for any of the seven to produce a recognizable label OR value, not
  just an unlucky preprocessing choice. The response's `warnings`
  makes this explicit per-request rather than silently returning
  `null`. The variants in `_KTP_VARIANT_SPECS` were chosen empirically
  against this one real sample — a differently-degraded photo (heavier
  blur, worse lighting, a different tilt, genuine motion blur a
  deconvolution kernel actually matches) could need different variants
  to cover its own weak spots; treat that list as a starting point to
  extend with more (test) real photos, not a fixed, finished set. Also
  worth knowing: `merge_ktp_results`'s confidence-then-cleanliness
  tie-break picks the more PLAUSIBLE-looking value across variants, not
  necessarily the CORRECT one — two variants agreeing on a wrong
  misread (both denoise settings smearing the same digit the same way,
  say) would still merge to that wrong value, same as any single-pass
  extraction can. It narrows the odds of a bad
  value passing through; it doesn't eliminate them, hence keeping
  every field an editable, reviewable value rather than a final answer.
- **`faktur_pajak/parser.py`'s field patterns were built and verified
  against two layouts** — a synthetic test invoice, and a real DJP
  Coretax-issued Faktur Pajak (confirmed against an actual production
  PDF, including its "DPP Nilai Lain" 11/12 calculation scheme). A
  third, differently-templated e-Faktur is still a realistic
  possibility (DJP doesn't guarantee one canonical layout) and may use
  different label phrasing, a different seller/buyer block order, or a
  differently-shaped item table — verify against real (test) exports
  before trusting this on a new source unseen so far, the same way the
  KTP parser needed tuning against real KTP photos. The multi-line
  item-block regex (`_LINE_ITEM_BLOCK_RE`, the real Coretax format) and
  the one-row-per-item regex (`_LINE_ITEM_TABLE_RE`, the older format)
  are each tried in turn — a genuinely new item layout (e.g. an
  additional printed column, or PPN broken out per line rather than
  just PPnBM) won't match either without extending one of them.
- **`/v1/locate-signature`'s native-PDF path assumes a ruled table**
  (straight horizontal/vertical rules around the signature cell, as in
  the Peruri template). A signature block laid out without visible
  borders (spacing/whitespace only) will fall back to a fixed-height
  guess above the name (`MAX_BOX_HEIGHT` in `document/signature_locator.py`),
  at reduced confidence — verify against a real sample of any new
  template before trusting the box unreviewed. The scanned-document
  fallback (`ocr_scanned`) additionally depends on the table rules
  being visually distinct enough for morphological line detection to
  find them; a faxed or heavily compressed scan may not have clean
  enough lines for this to work at all, in which case it degrades to
  the same fixed-height guess.
