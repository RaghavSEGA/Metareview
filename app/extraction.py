"""
Turning either (a) a producer-uploaded zip of saved review pages, or (b) a pasted review URL,
into clean article text ready for classification.

Path (a) never touches the network from this app — it's parsing files the producer already
saved from their own browser, which is exactly why it works for outlets that block bots.
Path (b) does make outbound HTTP requests from wherever this app is deployed; that's a normal
thing for a deployed service to do, but note it in your own review process — some outlets'
terms of service restrict automated access, same discussion as covered when this was scoped.
"""
import csv
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Optional

import trafilatura
from pypdf import PdfReader


@dataclass
class ReviewSource:
    key: str
    outlet: str
    score: Optional[float]
    date: Optional[str]
    url: Optional[str]
    raw_html: Optional[str]
    text: Optional[str] = None
    error: Optional[str] = None
    # Set post-classification from classify.classify_review()'s `review_language` field (see
    # streamlit_app.py) — None until a run has actually looked at the text. Kept on the source
    # object (not just the run result) so it survives back into the review table for display.
    language: Optional[str] = None
    # Platform this specific review covers (e.g. "PlayStation 5", "PC") — populated automatically
    # for Metacritic-sourced reviews (see app/metacritic.py), left None (and hand-editable in the
    # review table) for paste-URL/zip-upload sources, which have no platform signal of their own.
    # Feeds matrix.compute_platform_averages() for the report's per-platform Review Averages
    # breakdown — see app/report.py.
    platform: Optional[str] = None


# Everything below was tuned against 15 real PDF-extracted review texts, not synthetic
# examples — every quirk called out in a comment here is a false positive or a false negative
# actually observed in that batch, not a hypothetical.
#
# Scoring keywords that meaningfully raise confidence a nearby number IS the review's own score,
# as opposed to something else on the page entirely — see _LABELED_* below. Deliberately does
# NOT include "Rating" — a real review page had "RatingPEGI 16" (the game's PEGI AGE rating,
# not a review score) immediately after that exact label, and "Rating: <age>" is an extremely
# common, unrelated pattern on review pages (ESRB/PEGI ratings are almost always shown
# prominently). "Score" and "Verdict" don't have an equivalent age-rating-shaped collision.
_SCORE_KEYWORD = r"(?:Overall\s*Score|Our\s*Verdict|Final\s*Score|Verdict|Score|Nota|Note|Wertung|Punteggio)"
# Non-greedy, no linebreak (plain "." excludes "\n") — lets a keyword connect to its number
# through a short run of other text (an icon glyph, a colon, even a brand name like "Verdict
# PCGamesN 8/10") while still refusing to reach across a paragraph break into unrelated text.
_CONNECTOR = r".{0,20}?"

# The slash's surrounding whitespace deliberately excludes newlines ([ \t]* instead of \s*) —
# a real review PDF had a URL ending "...review-ps5/" immediately followed on the next line by
# an unrelated page-footer "5 of 14", and \s* let the two bridge into a bogus "5/5" (misread as
# a perfect score). A real "X/10"-style score is always one visual token; it never has a line
# break inside it.
#
# The trailing (?!\d) blocks a following DIGIT (so "8/100" isn't misread as "8/10" plus a stray
# "0"), but deliberately does NOT block a following LETTER the way a \b boundary would — a real
# review had "...Our VerdictPCGamesN 8/10Whether you're..." with no space before "Whether";
# pypdf's text extraction frequently loses inter-element whitespace, and a stricter boundary
# silently rejected this genuine, correctly-formatted score.
# The optional [ \t]? around the decimal separator (not just \d{1,2} directly after [.,])
# tolerates another real pypdf artifact: a decimal number rendered as separate text runs came
# out as "7 .3" (a space inserted between the integer part and the decimal point) — without
# this, the regex would skip past the "7" and wrongly start matching at the "3", silently
# producing a plausible-looking but wrong value ("3 out of 10" = 30, instead of "7.3 out of 10"
# = 73) rather than either the right number or no match at all.
_DECIMAL_NUM = r"\d{1,3}(?:[ \t]?[.,][ \t]?\d{1,2})?"
_FRACTION = r"(" + _DECIMAL_NUM + r")[ \t]*/[ \t]*(5|10|100)(?!\d)(?:(?![ \t]*/[ \t]*\d))"
_OUT_OF = r"(" + _DECIMAL_NUM + r")\s*out of\s*(5|10|100)(?!\d)"
_PERCENT = r"(\d{1,3})\s*%"
_BARE_INT = r"(\d{1,2})(?!\d)"

_LABELED_FRACTION_RE = re.compile(_SCORE_KEYWORD + _CONNECTOR + _FRACTION, re.IGNORECASE)
_OUT_OF_RE = re.compile(_OUT_OF, re.IGNORECASE)
_LABELED_PERCENT_RE = re.compile(_SCORE_KEYWORD + _CONNECTOR + _PERCENT, re.IGNORECASE)
_LABELED_BARE_INT_RE = re.compile(_SCORE_KEYWORD + _CONNECTOR + _BARE_INT, re.IGNORECASE)
_BARE_FRACTION_RE = re.compile(_FRACTION)


def _fraction_to_pct(numerator: str, denominator: str) -> Optional[float]:
    value = float(numerator.replace(" ", "").replace("\t", "").replace(",", "."))
    denom = float(denominator)
    pct = value / denom * 100
    return round(pct, 1) if 0 <= pct <= 100 else None


def extract_score_from_text(text: Optional[str]) -> Optional[float]:
    """
    Best-effort extraction of a review's OWN stated score from its text, normalized to a 0-100
    scale to match the convention the rest of this app uses (Metacritic-style). This is what
    backfills the Score column when a zip upload has no manifest.csv — previously it was left
    blank for the producer to fill in by hand for every single review.

    Five tiers, most confident first, each trying only if the previous found nothing usable:
      1. An explicit fraction near a scoring keyword — "Score: 8/10", "Our Verdict...8/10".
      2. "X out of N" phrasing anywhere — e.g. "7.3 out of 10" — distinctive enough on its own.
      3. A percent near a scoring keyword — "Score: 90%".
      4. A bare 1-2 digit integer directly after a scoring keyword with no slash or "%" at all
         (e.g. "Score: 10", or "Score" then a linebreak then "10") — ambiguous scale, so 1-10 is
         assumed out of 10 (by far the most common convention for a lone small integer) and
         11-100 is assumed already on a 0-100 scale.
      5. Any plain "X/10"-style fraction anywhere, unlabeled — lowest confidence, but still
         better than nothing. Checks every match in the text, not just the first, so an earlier
         fraction-shaped false positive doesn't block a later, real one from ever being reached.

    Deliberately NOT included: a bare, unlabeled percent anywhere in the text. Real review pages
    are full of unrelated percentages — ad copy ("86% OFF"), regional promos ("71% di sconto"),
    completion stats — and matching any of those as if it were the review's score is worse than
    finding nothing. Tier 3 above still catches a percent score, just only when a scoring
    keyword nearby actually supports it.

    Not a guarantee even with all five tiers: a genuinely prose-only review, or a score
    rendered as an icon-font glyph or an image rather than real text (both seen in this app's
    real-world test batch), won't match anything — and shouldn't. classify.py's `stated_score`
    field runs a second, LLM-based pass during classification that can catch a few more of
    these. Either way, the Score column in the review table stays hand-editable regardless.
    """
    if not text:
        return None

    m = _LABELED_FRACTION_RE.search(text)
    if m:
        pct = _fraction_to_pct(m.group(1), m.group(2))
        if pct is not None:
            return pct

    m = _OUT_OF_RE.search(text)
    if m:
        pct = _fraction_to_pct(m.group(1), m.group(2))
        if pct is not None:
            return pct

    m = _LABELED_PERCENT_RE.search(text)
    if m:
        pct = float(m.group(1))
        if 0 <= pct <= 100:
            return pct

    m = _LABELED_BARE_INT_RE.search(text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            return float(n * 10)
        if 11 <= n <= 100:
            return float(n)

    for m in _BARE_FRACTION_RE.finditer(text):
        pct = _fraction_to_pct(m.group(1), m.group(2))
        if pct is not None:
            return pct

    return None


def extract_text_from_html(html: str) -> Optional[str]:
    """Main-content extraction — strips nav/ads/boilerplate, keeps the article body."""
    if not html:
        return None
    return trafilatura.extract(html, include_comments=False, include_tables=False)


def extract_text_from_pdf(pdf_bytes: bytes):
    """
    Pulls page text directly out of a PDF (e.g. a saved/printed review page, or a scanned press
    clipping the producer exported to PDF). Unlike extract_text_from_html, there's no
    boilerplate-stripping pass here — a PDF export rarely carries the nav/ad/footer clutter a
    live web page does, so raw page text is normally already clean enough to classify.

    Returns (text, error) rather than raising — a scanned/image-only PDF with no text layer
    (pypdf can't OCR) should surface as a clear per-file error in the review table, not blow up
    the whole batch.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        text = "\n\n".join(p for p in pages if p)
        if not text:
            return None, "No extractable text found (likely a scanned/image-only PDF with no text layer)."
        return text, None
    except Exception as e:  # noqa: BLE001 - surface any parse failure to the UI, don't crash the batch
        return None, f"Failed to parse PDF: {e}"


def parse_uploaded_zip(zip_bytes: bytes, manifest_csv_bytes: Optional[bytes] = None):
    """
    Expects a zip of .html/.htm and/or .pdf files, any mix. If a manifest CSV is provided, it
    must have columns: filename, outlet, score, date [, url]. Without a manifest, outlet/score/
    date are left blank and must be filled in by the user in the review-UI table before running.
    """
    manifest = {}
    if manifest_csv_bytes:
        reader = csv.DictReader(io.StringIO(manifest_csv_bytes.decode("utf-8-sig")))
        for row in reader:
            manifest[row["filename"].strip()] = row

    sources = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".html", ".htm", ".pdf"))]
        for name in names:
            raw_bytes = zf.read(name)
            base = name.split("/")[-1]
            meta = manifest.get(base, {})
            key = base.rsplit(".", 1)[0]

            if name.lower().endswith(".pdf"):
                raw_html, (text, error) = None, extract_text_from_pdf(raw_bytes)
            else:
                raw_html = raw_bytes.decode("utf-8", errors="replace")
                text, error = extract_text_from_html(raw_html), None

            # Manifest score wins if given (it's an explicit human-provided value, including a
            # deliberate 0); otherwise fall back to whatever the regex heuristic finds in the
            # text itself, rather than leaving every review's score blank until hand-filled.
            manifest_score = _safe_float(meta.get("score"))
            score = manifest_score if manifest_score is not None else extract_score_from_text(text)

            sources.append(ReviewSource(
                key=key,
                # Fall back to the extension-stripped filename, not the raw filename — a
                # trailing ".pdf"/".html" is pure noise for curated-list matching and for the
                # outlet name shown in the report. It'll often still carry a game-title prefix
                # (e.g. "P3R Destructoid") without a manifest; is_on_curated_list() is built to
                # match through that, but the Outlet column stays hand-editable for the rest.
                outlet=meta.get("outlet", key),
                score=score,
                date=meta.get("date"),
                url=meta.get("url"),
                raw_html=raw_html,
                text=text,
                error=error,
            ))
    return sources


def fetch_review_url(fetch_fn, url: str, key: str, outlet_hint: str = "") -> ReviewSource:
    """
    `fetch_fn(url) -> str` is injected (e.g. `lambda u: requests.get(u, timeout=20).text`) so
    this stays testable without a live network call and so the network policy for outbound
    fetches lives in one obvious place the deploying team controls.
    """
    try:
        html = fetch_fn(url)
        text = extract_text_from_html(html)
        return ReviewSource(
            key=key, outlet=outlet_hint or url, score=extract_score_from_text(text), date=None,
            url=url, raw_html=html, text=text,
        )
    except Exception as e:  # noqa: BLE001 - surface any fetch failure to the UI, don't crash the run
        return ReviewSource(
            key=key, outlet=outlet_hint or url, score=None, date=None, url=url,
            raw_html=None, text=None, error=str(e),
        )


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None