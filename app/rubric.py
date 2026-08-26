"""
Shared constants: the sentiment rubric, starting category list, and curated-outlet list
handling. Keep this the single source of truth — matrix.py, classify.py, and report.py should
all import from here rather than re-defining rules, so a rubric change only has to happen once.
"""
import json
import re
import time
from pathlib import Path

INCLUSION_THRESHOLD = 0.25  # a category must be mentioned by >=25% of reviews to be reported
DEFAULT_MIN_REVIEWS = 20
DEFAULT_TARGET_REVIEWS = 30
MAX_REVIEWS = 100

DEFAULT_CATEGORIES = [
    "Battle/Combat Gameplay",
    "Exploration/Level Design",
    "Story/Narrative",
    "Characters",
    "Progression/RPG Systems",
    "Difficulty",
    "Visuals/Graphics",
    "UI/Menus",
    "Music/Sound",
    "Voice Acting",
    "Localization/Translation",
    "Performance/Technical",
    "Controls",
    "Content/Value",
]

CLASSIFICATION_SYSTEM_PROMPT = """You are applying Sega's metareview methodology to a single \
game review. Read the review text and classify it against the given category list.

Rubric (apply exactly):
- A critique only counts if the reviewer expresses an actual opinion (positive or negative) \
about that specific facet of the game. A mention with no opinion attached is EXCLUDED entirely \
— do not output a value for it at all, and do not call it neutral.
- value = 1 if the reviewer is positive about that facet.
- value = -1 if the reviewer is negative about that facet.
- value = 0 (neutral) ONLY if the SAME reviewer makes both a positive and a negative comment \
about the SAME facet in this review — the two offset. Do not use 0 for anything else.
- Only include a category in your output if the review actually contains a scorable critique \
for it (per the rules above). Omit categories the review doesn't address with an opinion.
- For every category you do output, include a short supporting quote (prefer exact wording \
from the review text you were given).
- Independently, list any recurring topic you notice that ISN'T on the given category list but \
that this review expresses a clear opinion about (e.g. a franchise-specific mechanic, a \
translation quality issue, a specific character). These become candidate emergent categories \
once tallied across the whole review set — that tally happens outside this call, so just \
report what you found in THIS review.

The review text may be in any language — classify it with the same rigor regardless of the \
source language; this is well within your fluency. Two things follow from that:
- The "quote" attached to every classification you output must always be in English. If the \
review itself isn't in English, translate the quote concisely and faithfully — preserve the \
original meaning exactly, never sanitize or embellish it, and never leave a quote untranslated \
in its source language. This report is compiled in English regardless of which outlets it draws \
from.
- Report `review_language`: the primary language the review is written in, as a plain English \
language name (e.g. "English", "French", "Japanese") — this flags for the producer which \
reviews' quotes passed through translation.

Also report the review's own stated score, if it gives one:
- `stated_score`: the review's explicit numeric score/rating, normalized to a 0-100 scale (e.g. \
"8.5/10" -> 85, "4 out of 5 stars" -> 80, "90%" -> 90, and the same for a non-English equivalent \
like "Note : 8/10" or "Nota: 8,5/10"). null if the review is prose-only with no stated score — \
never infer or invent a score from tone or from how positive/negative the review reads.
- `stated_score_raw`: the score exactly as it appeared in the review's own text (e.g. "8.5/10"), \
for producer verification. null if `stated_score` is null.
"""

_BUNDLED_LIST_PATH = Path(__file__).parent / "curated_outlets_snapshot.json"
CURATED_LIST_HELP_URL = (
    "https://metacritichelp.zendesk.com/hc/en-us/articles/"
    "14483198627607-Which-game-critics-and-publications-are-included-in-your-calculations"
)


def load_bundled_curated_list():
    """Fallback list, snapshotted when this app was built. Re-fetch live when possible —
    Metacritic revises this list several times a year; don't treat this snapshot as current."""
    with open(_BUNDLED_LIST_PATH) as f:
        payload = json.load(f)
    return payload["outlets"], payload["snapshot_date"]


def fetch_curated_list_live(fetch_fn):
    """
    `fetch_fn` is injected so this module has no direct network dependency (and so it's
    testable without hitting the network). Pass a function like:

        def fetch_fn(url: str) -> str:
            return requests.get(url, timeout=15).text

    and this will parse outlet names out of the page. Returns (outlets, fetched_at_epoch) or
    raises — caller should catch and fall back to `load_bundled_curated_list()`.
    """
    html = fetch_fn(CURATED_LIST_HELP_URL)
    # Metacritic's help page structure can change; this is a best-effort parse. If it breaks,
    # the bundled snapshot fallback keeps the app usable — just flag staleness in the UI.
    import re

    text_blocks = re.findall(r">([^<>]{2,60})<", html)
    # crude heuristic: outlet names are short, capitalized, not boilerplate sentences
    candidates = [
        t.strip() for t in text_blocks
        if t.strip() and not t.strip().endswith(".") and len(t.strip().split()) <= 6
    ]
    outlets = sorted(set(candidates))
    return outlets, time.time()


def _normalize_loose(s: str) -> str:
    """Lowercase, strip everything but letters/digits — for matching past purely cosmetic
    differences like "COG Connected" vs the curated list's "COGconnected"."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_on_curated_list(outlet_name: str, curated_outlets) -> bool:
    """
    Three passes, loosest last. A zip upload with no manifest.csv falls back to the filename
    as the outlet label (see extraction.parse_uploaded_zip) — that label often carries a game-
    title prefix ("P3R Destructoid") rather than being the bare outlet name, so an exact-match-
    only check would silently fail to match almost every real outlet and make "restrict to
    curated list" wrongly exclude everything.

      1. Exact match (case/whitespace insensitive) — the common case when a manifest.csv, or a
         hand-typed outlet name, gives the bare outlet name directly.
      2. Whole-word/phrase containment — matches a real outlet name embedded inside a longer
         filename-derived label (e.g. "Destructoid" inside "P3R Destructoid"). Word-bounded so
         a short name like "IGN" doesn't spuriously match inside an unrelated word.
      3. Punctuation/whitespace-insensitive containment, restricted to longer names only (>=6
         chars) to keep short-acronym false positives out — catches cosmetic-only differences
         like "COG Connected" vs "COGconnected".

    A miss doesn't necessarily mean the outlet isn't legitimate — it may genuinely not be on the
    current curated snapshot, or the filename may not resemble the real outlet name at all.
    Either way, the Outlet column and the on_curated_list checkbox in the review table are both
    editable by hand for exactly this reason.
    """
    norm = outlet_name.strip().lower()
    if any(norm == o.strip().lower() for o in curated_outlets):
        return True

    for o in curated_outlets:
        o_norm = o.strip().lower()
        if len(o_norm) >= 3 and re.search(r"\b" + re.escape(o_norm) + r"\b", norm):
            return True

    loose = _normalize_loose(outlet_name)
    for o in curated_outlets:
        o_loose = _normalize_loose(o)
        if len(o_loose) >= 6 and o_loose in loose:
            return True
    return False