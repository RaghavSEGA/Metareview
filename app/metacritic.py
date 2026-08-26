"""
Bulk-sources Metacritic critic reviews for a game: the review metadata (outlet, score, date,
review URL, pull-quote) via the JSON API the Metacritic page itself calls to page through
reviews as you scroll — no browser needed for this part — then the FULL review text from each
outlet's own page.

The metadata API was found by viewing source on a Metacritic critic-reviews page and finding
the embedded Nuxt payload that records the exact request the page's own JS makes:

    https://backend.metacritic.com/reviews/metacritic/critic/games/<slug>/web
        ?offset=0&limit=10&filterBySentiment=all&sort=score
        &componentName=critic-reviews&componentDisplayName=critic+Reviews&componentType=ReviewList

It's plain JSON, no auth/cookies needed, and returns 10 reviews per call regardless of the
requested `limit` — so fetch_review_list() pages through `offset` in steps of 10 until a page
comes back empty. Note this is Metacritic's current site build; if they ship a redesign this
endpoint could move, but the same view-source trick will find wherever it moved to.

Filtering by platform is NOT a query param on this URL — an earlier version of this module
assumed `?platform=<slug>` would filter the results (it's what the *page* URL uses, e.g.
metacritic.com/game/<slug>/critic-reviews/?platform=xbox-series-x) and shipped without ever
actually confirming it against a live response; it silently has no effect on the API and always
returns the lead platform's reviews regardless of what's passed. Confirmed directly against
Metacritic's real API responses (fetched via a tool that isn't subject to this sandbox's
network allowlist) for a real multi-platform title (Persona 3 Reload) — passing
`?platform=xbox-series-x`, `?platform=1500000129` (the platform's numeric id), and
`?platformId=1500000129` on the URL above all returned the identical PlayStation 5 result set.
The correct filter is a URL PATH segment instead, found the same way — fetching
metacritic.com/game/<slug>/critic-reviews/?platform=xbox-series-x directly and confirming its
rendered content is genuinely Xbox-specific, then locating the real API shape from there:

    https://backend.metacritic.com/reviews/metacritic/critic/games/<slug>/platform/<platform>/web
        ?offset=0&limit=10&filterBySentiment=all&sort=score
        &componentName=critic-reviews&componentDisplayName=critic+Reviews&componentType=ReviewList

— i.e. `/platform/<platform-slug>/` inserted right before the trailing `/web`. Verified this
actually filters (distinct, correct outlet lists for xbox-series-x and pc, neither matching the
lead-platform PS5 set) and that offset pagination still walks forward correctly within a single
platform's results.

Full review TEXT is a separate problem: Metacritic's own pull-quote is usually one sentence,
nowhere near the full article a producer needs to classify. So for each review's outlet URL,
this module tries two tiers:
  1. A plain requests.get() + trafilatura extraction (extraction.extract_text_from_html) —
     fast, no browser runtime needed, and sufficient for any outlet that server-renders its
     article body (most of them).
  2. A headless-browser (Playwright) render, used ONLY as a fallback when tier 1 comes back
     empty — either because the outlet renders its article body client-side with JS, or
     because it returned a bot-check/empty page to a bare requests call.

Deployment note: tier 2 needs Chromium available via Playwright (`playwright install
--with-deps chromium`) — see Dockerfile. If Playwright or its browser isn't installed in a
given deployment, tier 1 still works for every outlet it can handle on its own; tier 2 fetches
simply report a clear per-review error instead of crashing the batch, exactly like any other
fetch failure this app already tolerates (see extraction.fetch_review_url).

On a host with no shell/build-hook access — Streamlit Community Cloud, notably — there's no way
to run `playwright install chromium` as a separate step the way the Dockerfile and README's
"Local run" do: `pip install -r requirements.txt` installs the `playwright` Python package but
never the browser binary itself. _ensure_playwright_chromium_installed() below covers this by
installing the browser lazily, on the first Playwright launch attempt that actually fails for
that specific reason, rather than assuming it's already present. See README.md's "Streamlit
Community Cloud" deployment section for the packages.txt this still requires (the OS-level
libraries Chromium itself needs — no amount of Python-side installing substitutes for those).
"""
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Callable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests

from .extraction import ReviewSource, extract_text_from_html

_LIST_API_URL = "https://backend.metacritic.com/reviews/metacritic/critic/games/{slug}/web"
_LIST_API_URL_FOR_PLATFORM = (
    "https://backend.metacritic.com/reviews/metacritic/critic/games/{slug}/platform/{platform}/web"
)
_GAME_API_URL = "https://backend.metacritic.com/games/metacritic/{slug}/web"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.metacritic.com/",
    "Accept": "application/json",
}


def parse_game_url_or_slug(value: str) -> Tuple[str, Optional[str]]:
    """
    Accepts either a bare slug ("persona-3-reload") or a full Metacritic critic-reviews URL
    ("https://www.metacritic.com/game/persona-3-reload/critic-reviews/?platform=playstation-5")
    and returns (slug, platform) — platform is None unless the URL specified one.
    """
    value = value.strip()
    m = re.search(r"metacritic\.com/game/([^/?#]+)", value)
    if not m:
        return value, None
    slug = m.group(1)
    platform = parse_qs(urlparse(value).query).get("platform", [None])[0]
    return slug, platform


def fetch_review_list(slug: str, platform: Optional[str] = None, page_size: int = 10,
                       max_pages: int = 50, pause_s: float = 0.4) -> List[dict]:
    """
    Returns the raw list of review-metadata dicts from Metacritic's own API (each has at
    least: url, publicationName, score, date, quote, platform). Pages through `offset` until
    an empty page comes back, rather than trusting any single "total" figure — robust
    regardless of how a platform filter changes the true count.

    IMPORTANT: leaving `platform` as None does NOT return reviews for every platform release —
    confirmed empirically against a real multi-platform title (Persona 3 Reload: 6 platforms).
    It returns only the single "lead platform" Metacritic's own page defaults to showing (the
    one flagged isLeadPlatform=true in the game's platform list). To consolidate reviews across
    every platform a game released on, use fetch_review_list_all_platforms() /
    build_sources_from_metacritic_all_platforms() instead of looping this function yourself.

    `platform`, when given, must be the platform's own slug (e.g. "xbox-series-x", "pc" — see
    fetch_game_platforms()'s `slug` field), and is applied as a `/platform/<slug>/` URL PATH
    segment (_LIST_API_URL_FOR_PLATFORM), NOT a query parameter — see this module's docstring
    for how that was confirmed against live API responses. Passing a platform slug that doesn't
    exist for this game is indistinguishable from that platform genuinely having zero reviews:
    both just return an empty first page, same as any other empty result.

    max_pages is a hard backstop (500 reviews) against an unexpected API change causing an
    infinite loop, not a real-world limit — no game has anywhere near that many critic reviews.
    """
    items: List[dict] = []
    offset = 0
    session = requests.Session()
    session.headers.update(_HEADERS)
    url = (_LIST_API_URL_FOR_PLATFORM.format(slug=slug, platform=platform) if platform
           else _LIST_API_URL.format(slug=slug))
    for _ in range(max_pages):
        params = {
            "offset": offset, "limit": page_size, "filterBySentiment": "all", "sort": "score",
            "componentName": "critic-reviews", "componentDisplayName": "critic Reviews",
            "componentType": "ReviewList",
        }
        resp = session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        page_items = resp.json().get("data", {}).get("items", [])
        if not page_items:
            break
        items.extend(page_items)
        offset += len(page_items)
        time.sleep(pause_s)  # be polite between requests — this is someone else's API, not ours
    return items


def fetch_game_platforms(slug: str) -> List[dict]:
    """
    Returns every platform this game was released on: [{"name", "slug", "review_count"}, ...].
    Found the same way as the reviews-list API — viewing source on a Metacritic game page and
    reading its embedded Nuxt payload, which records the "self" link for the product API call
    that filled in the page (.../games/metacritic/<slug>/web?componentName=product&...) and the
    "platforms" array it returns, each entry carrying its own criticScoreSummary.reviewCount
    and platform slug (e.g. "playstation-5", "pc", "xbox-series-x").

    This is what makes cross-platform consolidation possible: fetch_review_list() alone can
    only pull one platform at a time, and doesn't know what platforms exist to loop over.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    params = {"componentName": "product", "componentDisplayName": "Product",
              "componentType": "Product"}
    resp = session.get(_GAME_API_URL.format(slug=slug), params=params, timeout=20)
    resp.raise_for_status()
    item = resp.json().get("data", {}).get("item", {})
    platforms = []
    for p in item.get("platforms", []):
        summary = p.get("criticScoreSummary") or {}
        platforms.append({
            "name": p.get("name"),
            "slug": p.get("slug"),
            "review_count": summary.get("reviewCount") or 0,
        })
    return platforms


def fetch_review_list_all_platforms(slug: str) -> List[dict]:
    """
    Consolidates review metadata across EVERY platform the game released on, rather than just
    the single "lead platform" fetch_review_list() returns when no `platform` is given (see its
    docstring) — this is the actual fix for "I don't want to have to go platform by platform".

    Two real-world wrinkles handled here, both confirmed against real multi-platform titles:
      1. The exact same review URL can appear under more than one platform's list (an outlet
         that published one review covering multiple platform versions at once) — deduplicated
         by URL so it isn't double-counted in the matrix.
      2. The same outlet can also publish GENUINELY SEPARATE reviews for different platform
         versions of the same game (e.g. a PS5 review and a distinct PC review, different
         scores and text) — both are real, independent reviews and both are kept. To keep the
         review table from showing two identically-labeled rows in that case, the platform name
         is appended to the outlet label, but ONLY for outlets that actually appear more than
         once post-dedupe — a single-platform outlet's label is left untouched.
    """
    reviewable = [p for p in fetch_game_platforms(slug) if p["review_count"] > 0]

    all_items = []
    for p in reviewable:
        for item in fetch_review_list(slug, platform=p["slug"]):
            item["_platform_name"] = p["name"]
            all_items.append(item)

    seen_urls = set()
    deduped = []
    for item in all_items:
        url = item.get("url")
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        deduped.append(item)

    outlet_counts = Counter(item.get("publicationName") for item in deduped)
    for item in deduped:
        name = item.get("publicationName")
        if name and outlet_counts[name] > 1:
            item["publicationName"] = f"{name} ({item.get('_platform_name') or 'unknown platform'})"

    return deduped


def _try_requests(url: str, timeout: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return extract_text_from_html(resp.text), None
    except Exception as e:  # noqa: BLE001 - any failure here just means "try tier 2 next"
        return None, str(e)


_playwright_install_state = {"attempted": False, "error": None}


def _ensure_playwright_chromium_installed() -> Optional[str]:
    """
    On a host with no shell/build-hook access — Streamlit Community Cloud, notably —
    `pip install -r requirements.txt` installs the `playwright` Python package but never
    downloads the actual Chromium binary; that's normally a separate `playwright install
    chromium` step (Dockerfile, README's "Local run") that Community Cloud's build process has
    nowhere to run. This installs it lazily instead, called from _try_playwright() only once its
    first launch attempt has already failed with Playwright's specific "browser not installed"
    error — so a deployment where the browser WAS pre-installed (Docker, a plain host that ran
    the README's setup step) never pays this cost at all.

    Runs the actual install subprocess at most once per process: if it fails (no disk space, no
    network egress to Playwright's CDN, etc.), every subsequent Playwright fallback in this same
    run reuses that one failure immediately rather than each retrying its own slow, doomed
    install call and dragging out the whole batch behind it.

    Returns None on success (or if already attempted and succeeded), or an error string on
    failure — the caller folds this into the same per-review error reporting as any other
    Playwright failure.
    """
    if _playwright_install_state["attempted"]:
        return _playwright_install_state["error"]
    _playwright_install_state["attempted"] = True
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True, capture_output=True, text=True, timeout=300,
        )
        return None
    except subprocess.CalledProcessError as e:
        # This can be a wall of download-progress output; the last few stderr lines are what
        # actually explains the failure to a producer reading the review table's error column
        # (e.g. a missing OS-level library — see README's "Streamlit Community Cloud" section
        # for the packages.txt this doesn't substitute for).
        tail = " | ".join((e.stderr or "").strip().splitlines()[-5:])
        error = f"playwright install chromium failed: {tail or e}"
    except Exception as e:  # noqa: BLE001 - covers subprocess timeout, missing python, etc.
        error = f"playwright install chromium failed: {e}"
    _playwright_install_state["error"] = error
    return error


def _try_playwright(url: str, timeout: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, ("Playwright not installed in this environment — "
                       "pip install playwright && playwright install chromium")
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as launch_err:
                if "Executable doesn't exist" not in str(launch_err):
                    raise
                install_error = _ensure_playwright_chromium_installed()
                if install_error:
                    return None, install_error
                browser = p.chromium.launch(headless=True)  # retry now that it's installed
            page = browser.new_page(user_agent=_HEADERS["User-Agent"])
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_timeout(1500)  # let client-side-rendered article bodies finish paint
            html = page.content()
            browser.close()
        return extract_text_from_html(html), None
    except Exception as e:  # noqa: BLE001 - surfaced to the caller as this review's error
        return None, str(e)


def fetch_full_text(url: str, timeout: int = 20) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns (text, method, error). method is "requests" or "playwright" — whichever tier
    actually produced usable text. error is None on success; on failure it's a combined
    message covering why BOTH tiers came up empty (so a producer isn't left guessing whether
    the outlet blocked the request, has no article-body text at all, or Playwright simply
    isn't installed in this deployment).
    """
    text, err1 = _try_requests(url, timeout)
    if text:
        return text, "requests", None

    text2, err2 = _try_playwright(url, timeout)
    if text2:
        return text2, "playwright", None

    combined = (f"requests: {err1 or 'no extractable article text'}; "
                f"playwright: {err2 or 'no extractable article text'}")
    return None, "none", combined


def _make_key(index: int, outlet: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", outlet.lower()).strip("-") or "review"
    return f"mc_{index}_{slug}"


def _sources_from_items(
    items: List[dict], progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> List[ReviewSource]:
    """
    Shared by both build_sources_from_metacritic() and its all-platforms sibling: given a list
    of Metacritic review-metadata dicts, fetches each one's full text (requests first,
    Playwright fallback) and returns one ReviewSource per review — same shape the zip-upload
    and paste-URL paths already produce, so this plugs into the existing review table /
    classification pipeline unchanged.

    progress_cb(done, total, outlet), if given, is called after each review is fetched — used
    by the Streamlit UI to drive a progress bar, since the full-text fetch (especially any
    Playwright fallbacks) is the slow part of this, not the metadata list call(s).
    """
    total = len(items)
    sources = []
    for i, item in enumerate(items):
        url = item.get("url")
        outlet = item.get("publicationName") or url or f"Review {i + 1}"
        score = item.get("score")  # already 0-100 scale — matches this app's convention
        date = item.get("date")
        # "platform" is Metacritic's own field on each item; "_platform_name" is the friendlier
        # name fetch_review_list_all_platforms() tags on (see its docstring) — prefer that when
        # both are present since it's what the outlet-disambiguation suffix uses too, so a
        # report's per-platform breakdown lines up with what the outlet labels already show.
        platform = item.get("_platform_name") or item.get("platform")

        if not url:
            text, error = None, "Metacritic did not provide a review URL for this entry."
        else:
            text, _method, error = fetch_full_text(url)

        sources.append(ReviewSource(
            key=_make_key(i, outlet), outlet=outlet, score=score, date=date, url=url,
            raw_html=None, text=text, error=error, platform=platform,
        ))
        if progress_cb:
            progress_cb(i + 1, total, outlet)
    return sources


def build_sources_from_metacritic(
    slug: str, platform: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> List[ReviewSource]:
    """
    Single-platform pipeline: fetch the review list for ONE platform (or Metacritic's default
    "lead platform" if `platform` is left None — see fetch_review_list()'s docstring) and
    build a ReviewSource per review. Use build_sources_from_metacritic_all_platforms() instead
    when the goal is a consolidated metareview across every platform release, which is the
    common case for a company-wide report.
    """
    items = fetch_review_list(slug, platform=platform)
    return _sources_from_items(items, progress_cb)


def build_sources_from_metacritic_all_platforms(
    slug: str, progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> List[ReviewSource]:
    """
    Consolidated pipeline: discovers every platform the game released on, pulls and merges
    review metadata from all of them (fetch_review_list_all_platforms — see its docstring for
    the dedup/disambiguation rules), then fetches full text and builds a ReviewSource per
    review exactly like build_sources_from_metacritic(). This is the "don't make me go
    platform by platform" entry point.
    """
    items = fetch_review_list_all_platforms(slug)
    return _sources_from_items(items, progress_cb)
