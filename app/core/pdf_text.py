"""
Native-PDF text + vector-line extraction via PyMuPDF.

Used only when the input file is a real, digitally generated PDF (a
CompliFi-issued form, a Peruri template, an e-signed contract) rather
than a scanned/photographed one. On these, every character and every
table rule is already stored as an object in the PDF at exact point
coordinates - so locating "the box directly above this printed name"
is a geometry lookup, not a computer-vision problem. See
document/signature_locator.py for why this matters and when the OCR/pixel-based
fallback kicks in instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass(frozen=True)
class TextLine:
    page_index: int
    text: str
    rect: fitz.Rect  # bounding box of the whole line, in PDF points (top-left origin)


@dataclass(frozen=True)
class RuleLine:
    """One straight line segment recovered from the page's vector
    graphics - almost always a table border/rule in a form like this
    one, never body-text glyphs (those are handled separately as
    TextLine)."""

    orientation: str  # "h" or "v"
    pos: float  # the shared coordinate: y for a horizontal rule, x for a vertical one
    start: float  # the other coordinate's start
    end: float  # the other coordinate's end


MIN_RULE_SPAN = 5.0  # ignore tiny serifs/decoration - not real table rules
# (kept low deliberately: a single-line row's own border segment - e.g.
# the ~10pt-tall vertical rule beside a one-line name row, as opposed
# to the taller multi-line blank cell above it - is still short. Text
# glyphs in a normal PDF are drawn via font show operators, not filled
# rectangles, so this threshold isn't at risk of picking up letterforms
# - only genuinely drawn rules/borders show up as thin filled "re"/"l"
# shapes at all.)
MAX_RULE_THICKNESS = 2.0  # how "thin" a shape must be to count as a line rather than a filled block


def has_extractable_text(doc: fitz.Document, min_chars: int = 20) -> bool:
    """True if the PDF carries real text objects rather than being a
    scan/photo wrapped in a PDF container (which yields ~0 characters
    of native text). Callers use this to decide between the native
    geometry lookup and the OCR + pixel-line fallback."""
    total = 0
    for page in doc:
        total += len(page.get_text("text"))
        if total >= min_chars:
            return True
    return False


def extract_full_text(doc: fitz.Document) -> str:
    """All pages' native text, in reading order, one page per block.

    Used for a digitally generated document (a Coretax/e-Faktur PDF,
    any CompliFi-issued form) where the label->value text parsers
    (e.g. faktur_pajak/parser.py) can work directly off PyMuPDF's own text
    extraction instead of a raster + Tesseract round trip - faster and
    far more accurate than OCR on text that was never an image to
    begin with. Gate this on has_extractable_text() first; a scanned
    PDF returns ~nothing useful here.
    """
    return "\n".join(page.get_text("text") for page in doc)


def extract_lines(page: fitz.Page) -> list[TextLine]:
    """Groups words into their printed lines (PyMuPDF's own block/line
    grouping) and returns each line's full text plus bounding box.

    A signer's name is matched as a LINE, not a single word - "Muhammad
    Gifar Z." is three separate word objects in the PDF that happen to
    share a line, and only the line as a whole is a meaningful match
    target.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)
    grouped: dict[tuple[int, int], list] = {}
    for x0, y0, x1, y1, text, block_no, line_no, _word_no in words:
        grouped.setdefault((block_no, line_no), []).append((x0, y0, x1, y1, text))

    lines: list[TextLine] = []
    for items in grouped.values():
        items.sort(key=lambda it: it[0])  # left-to-right reading order
        text = " ".join(it[4] for it in items)
        x0 = min(it[0] for it in items)
        y0 = min(it[1] for it in items)
        x1 = max(it[2] for it in items)
        y1 = max(it[3] for it in items)
        lines.append(TextLine(page_index=page.number, text=text, rect=fitz.Rect(x0, y0, x1, y1)))
    return lines


def extract_rule_lines(page: fitz.Page) -> list[RuleLine]:
    """Finds table borders/rules drawn as vector graphics on the page.

    Form templates draw table borders either as very thin FILLED
    rectangles (Peruri's own template does this - confirmed by
    inspecting its signature table, which is a grid of ~0.5pt-tall
    filled rects) or as STROKED line segments. This handles both by
    treating anything with near-zero width or height as a straight
    line rather than special-casing one drawing style, and by walking
    stroked path items ("l" = line-to) for the same shape when a
    drawing isn't already a thin rect.
    """
    rules: list[RuleLine] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        width, height = rect.x1 - rect.x0, rect.y1 - rect.y0
        if height <= MAX_RULE_THICKNESS and width >= MIN_RULE_SPAN:
            rules.append(RuleLine("h", (rect.y0 + rect.y1) / 2, rect.x0, rect.x1))
        elif width <= MAX_RULE_THICKNESS and height >= MIN_RULE_SPAN:
            rules.append(RuleLine("v", (rect.x0 + rect.x1) / 2, rect.y0, rect.y1))
        else:
            for item in drawing.get("items", []):
                if item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= MAX_RULE_THICKNESS and abs(p1.x - p2.x) >= MIN_RULE_SPAN:
                    rules.append(RuleLine("h", (p1.y + p2.y) / 2, min(p1.x, p2.x), max(p1.x, p2.x)))
                elif abs(p1.x - p2.x) <= MAX_RULE_THICKNESS and abs(p1.y - p2.y) >= MIN_RULE_SPAN:
                    rules.append(RuleLine("v", (p1.x + p2.x) / 2, min(p1.y, p2.y), max(p1.y, p2.y)))
    return rules
