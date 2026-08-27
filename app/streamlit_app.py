"""
Metareview Tool — Streamlit app.

Run locally:   streamlit run app/streamlit_app.py
See README.md for deployment notes (secrets, OTP vs SSO, hosting).
"""
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `from app import ...` when run directly

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app import auth, classify, metacritic, narrative
from app.extraction import fetch_review_url, parse_uploaded_zip
from app.matrix import build_matrix_workbook, compute_platform_averages, compute_weighted_scores
from app.report import build_report_docx
from app.rubric import (
    DEFAULT_CATEGORIES, DEFAULT_MIN_REVIEWS, DEFAULT_TARGET_REVIEWS, MAX_REVIEWS,
    INCLUSION_THRESHOLD, is_on_curated_list, load_bundled_curated_list,
)

st.set_page_config(page_title="Metareview Tool", page_icon="📊", layout="wide",
                    initial_sidebar_state="expanded")

# ── SEGA dark-theme CSS — matches the look of Sega's other internal Streamlit tools ────────
st.markdown("""
<style>
:root{--bg0:#040A1C;--bg1:#0D1B2A;--bg2:#0A1628;--bdr:#1A2A4A;--txt:#E0E0E0;
--mut:#8899BB;--blu:#00AADD;--gld:#F5C218;--grn:#00BB66;--red:#CC2244;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:var(--bg0)!important;color:var(--txt);}
[data-testid="stSidebar"]{background:var(--bg2)!important;}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
header[data-testid="stHeader"]{background:transparent!important;}footer{visibility:hidden;}
h1,h2,h3,h4,h5,h6{color:#fff!important;}
.mrv-card{background:var(--bg1);border:1px solid var(--bdr);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
.hero-t{font-size:2.2rem;font-weight:700;color:#fff;margin-bottom:0;}
.hero-s{font-size:1rem;color:var(--mut);margin-top:2px;padding-bottom:10px;border-bottom:2px solid transparent;
  border-image:linear-gradient(90deg,var(--blu),var(--gld) 45%,transparent 70%) 1;}
.stButton>button{background:var(--blu)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;transition:all .2s!important;}
.stButton>button:hover{background:#0088BB!important;transform:translateY(-1px);}
.stDownloadButton>button{background:transparent!important;border:1px solid var(--blu)!important;color:var(--blu)!important;border-radius:8px!important;font-weight:600!important;}
.stDownloadButton>button:hover{background:rgba(0,170,221,.12)!important;transform:translateY(-1px);}
textarea{background:var(--bg1)!important;color:var(--txt)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;}
textarea:focus,input:focus{border-color:var(--blu)!important;box-shadow:0 0 0 1px var(--blu)!important;}
[data-testid="stExpander"]{border:1px solid var(--bdr)!important;border-radius:10px!important;background:var(--bg1)!important;}
[data-testid="stExpander"]:hover{border-color:var(--blu)!important;}
[data-testid="stFileUploaderDropzone"]{background:var(--bg1)!important;border:1px dashed var(--bdr)!important;border-radius:10px!important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--blu)!important;}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--bdr);}
.stTabs [data-baseweb="tab"]{color:var(--mut);font-weight:600;padding:8px 18px;border-radius:8px 8px 0 0;background:transparent;}
.stTabs [aria-selected="true"]{color:var(--blu)!important;}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--blu);height:2px;}
[data-testid="stWidgetLabel"] p{color:var(--mut)!important;font-size:.82rem!important;font-weight:600!important;}
*::-webkit-scrollbar{width:10px;height:10px;}
*::-webkit-scrollbar-track{background:var(--bg0);}
*::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:5px;}
*::-webkit-scrollbar-thumb:hover{background:var(--blu);}
@media (prefers-reduced-motion: reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important;}}
</style>""", unsafe_allow_html=True)

auth.require_login()
auth.logout_button()

st.markdown(
    '<div class="hero-t">📊 Metareview Tool</div>'
    '<div class="hero-s">AI-assisted press-review sentiment matrix + draft report</div><br/>',
    unsafe_allow_html=True,
)
st.caption(
    "Validated once (Persona 3: Reload POC) — review and correct before treating output as "
    "final. Agents advise, humans decide."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0;">'
        '<span style="font-size:2.5rem;">📊</span><br/>'
        '<strong style="font-size:1.3rem;color:#fff;">Metareview Tool</strong><br/>'
        '<span style="color:#8899BB;font-size:.85rem;">v0.1 — proof-of-concept grade</span>'
        '</div>', unsafe_allow_html=True,
    )
    st.divider()

    client = classify.get_client()
    backend = "Not configured"
    if client is not None:
        backend = "AWS Bedrock" if type(client).__name__ == "AnthropicBedrock" else "Anthropic API"
    st.markdown(
        f'<div style="font-size:.8rem;color:#8899BB;">Model<br/>'
        f'<strong style="color:#00AADD;">Claude Sonnet 4.6</strong><br/>'
        f'<span style="color:#556;">via {backend}</span></div>', unsafe_allow_html=True,
    )
    if client is None:
        st.error("No LLM credentials configured — set AWS_BEDROCK_* or ANTHROPIC_API_KEY in secrets.")
    model = classify.DEFAULT_MODEL

    st.divider()
    target_reviews = st.slider("Target review count", DEFAULT_MIN_REVIEWS, MAX_REVIEWS,
                                DEFAULT_TARGET_REVIEWS)
    restrict_curated = st.checkbox("Restrict to Metacritic's curated critics list", value=True)
    curated_outlets, snapshot_date = load_bundled_curated_list()
    st.caption(f"Curated list snapshot: {snapshot_date} ({len(curated_outlets)} outlets). "
               "Re-fetch live before a real run if it's been a while — see rubric.py.")

    st.divider()
    parallel_workers = st.slider(
        "Parallel requests", min_value=1, max_value=10, value=5,
        help="How many reviews to fetch or classify at once instead of strictly one at a time — "
             "applies to the Metacritic full-text fetch and to both classification passes. "
             "Higher is faster on a large batch but raises the chance of tripping the LLM API's "
             "rate limit, and (Metacritic tab only) can spike memory if several reviews fall "
             "back to a concurrent headless-browser render at once. Set to 1 to go back to "
             "strictly sequential.",
    )

    st.divider()
    include_recommendations = st.checkbox(
        "Include \"PD Recommendations for Next Title\" section", value=True,
        help="Uncheck to leave this section out of the report entirely for this run — e.g. if "
             "recommendations are being handled elsewhere in your process for this title. The "
             "model isn't asked to draft any when this is off, and the section (and its "
             "heading) won't appear in the docx at all rather than showing up empty.",
    )

    st.divider()
    fan_reaction_notes = st.text_area(
        "Fan/user reaction notes (optional)",
        value="", height=100,
        placeholder="Paste raw notes from forums, storefront user reviews, surveys, etc. Leave "
                    "blank to skip this section entirely.",
        help="Adds an \"Impression of Fan Reaction\" section to the report, synthesized from "
             "whatever's pasted here — explicitly flagged in the report as impressionistic "
             "rather than scored data, unlike the press-review sections. This is raw notes for "
             "the model to summarize, not another batch of reviews to classify; leave it blank "
             "and the section (and its heading) is left out of the report entirely, the same "
             "way PD Recommendations is when that toggle is off.",
    )

# Applies a Metacritic game-info lookup queued by the "Look up game info from this link" button
# in the Metacritic tab (further down this script). This MUST run before the Game title/
# Platform(s)/Release date widgets below are created: Streamlit raises a StreamlitAPIException
# if you write to st.session_state[key] for a widget's key after that widget has already been
# instantiated in the current script run, so the button handler can't set mrv_game_title etc.
# directly on the same run it fires in — it stashes the fetched values under this neutral,
# non-widget-bound key instead and reruns; this block, running ahead of widget creation on
# every run, is what actually applies them.
_pending_game_info = st.session_state.pop("_mc_pending_game_info", None)
if _pending_game_info:
    if _pending_game_info.get("title"):
        st.session_state["mrv_game_title"] = _pending_game_info["title"]
    if _pending_game_info.get("platforms_text"):
        st.session_state["mrv_platforms"] = _pending_game_info["platforms_text"]
    if _pending_game_info.get("release_date"):
        st.session_state["mrv_release_date"] = _pending_game_info["release_date"]

# ---------------------------------------------------------------------------
# Game metadata
# ---------------------------------------------------------------------------
st.markdown('<div class="mrv-card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
# Explicit keys so the "Fetch from Metacritic" tab's game-info lookup can fill these in — see
# the pending-info block above and the "Look up game info from this link" button below.
game_title = col1.text_input("Game title", key="mrv_game_title")
platforms = col2.text_input("Platform(s)", key="mrv_platforms")
release_date = col3.date_input("Release date", value=None, key="mrv_release_date")

st.markdown(
    '<div style="font-size:.82rem;font-weight:600;color:#8899BB;margin-bottom:2px;">'
    "Categories (the axes reviews are scored against)</div>", unsafe_allow_html=True,
)
default_selected = st.multiselect(
    "Standard categories — uncheck any that don't apply to this title",
    options=DEFAULT_CATEGORIES, default=DEFAULT_CATEGORIES, label_visibility="collapsed",
)
custom_categories_text = st.text_area(
    "Custom categories for this title (one per line)",
    value="", height=90, placeholder="e.g. Recasting\nCombining Puyo Puyo and Tetris\nGacha/Monetization",
    help="Genre- or franchise-specific axes that aren't in the standard list. These get "
         "classified and scored exactly like the standard categories — add as many as this "
         "title needs. The classifier can still surface further emergent categories on its own "
         "from review text even beyond what you list here.",
)
custom_categories = [c.strip() for c in custom_categories_text.splitlines() if c.strip()]

# Union, de-duplicated case-insensitively, defaults first then custom, preserving input order.
categories, _seen = [], set()
for c in default_selected + custom_categories:
    key = c.lower()
    if key not in _seen:
        _seen.add(key)
        categories.append(c)

st.caption(f"{len(categories)} categor{'y' if len(categories) == 1 else 'ies'} will be scored "
           f"this run ({len(default_selected)} standard, {len(custom_categories)} custom).")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Review sourcing
# ---------------------------------------------------------------------------
st.subheader("Reviews")
tab_metacritic, tab_urls, tab_upload = st.tabs(
    ["Fetch from Metacritic", "Paste review URLs", "Upload saved HTML (zip)"]
)

if "sources" not in st.session_state:
    st.session_state["sources"] = []

with tab_metacritic:
    st.caption(
        "Pulls the critic-review list straight from Metacritic's own API (outlet, score, "
        "date, review URL, and pull-quote for every review — no scrolling or page limits), "
        "then fetches each outlet's FULL review text: a plain fetch first, falling back to a "
        "headless browser for outlets that render their article body with JavaScript or block "
        "a bare request outright. Metacritic's score is already on this app's 0-100 scale. "
        "By default this consolidates reviews across EVERY platform the game released on "
        "(not just whichever one Metacritic's own page shows first) — an outlet that reviewed "
        "both the PS5 and PC versions separately will show up as two rows, labeled by platform. "
        "Some outlets will still fail to fetch (hard paywalls, aggressive bot-blocking) — check "
        "the error column in the review table below and fall back to the upload tab, or paste "
        "the URL in the first tab, for any that don't come through."
    )
    mc_input = st.text_input(
        "Metacritic game URL or slug",
        placeholder="https://www.metacritic.com/game/persona-3-reload/critic-reviews/  "
                    "(or just: persona-3-reload)",
    )
    if st.button("Look up game info from this link") and mc_input.strip():
        _mc_lookup_slug, _ = metacritic.parse_game_url_or_slug(mc_input)
        try:
            _game_info = metacritic.fetch_game_info(_mc_lookup_slug)
        except Exception as e:  # noqa: BLE001 - surface a clear error, don't crash the app
            st.error(f"Couldn't look up game info: {e}")
        else:
            # Can't write mrv_game_title/mrv_platforms/mrv_release_date directly here — those
            # widgets were already created earlier in THIS run (the card sits above the tabs),
            # and Streamlit forbids setting a widget-bound session_state key after that widget
            # has run. Queue the values under a neutral key instead; the pending-info block near
            # the top of the script applies them on the rerun below, before those widgets exist
            # for that run.
            _filled = []
            _pending = {}
            if _game_info["title"]:
                _pending["title"] = _game_info["title"]
                _filled.append("title")
            if _game_info["platforms"]:
                # Also seeds mc_platforms (below) so the "restrict to one platform" picker has
                # its options ready without a second, separate lookup click. Not widget-bound,
                # so this one's safe to set directly.
                _pending["platforms_text"] = ", ".join(
                    p["name"] for p in _game_info["platforms"] if p.get("name")
                )
                st.session_state["mc_platforms"] = _game_info["platforms"]
                _filled.append("platform(s)")
            if _game_info["release_date"]:
                _pending["release_date"] = _game_info["release_date"]
                _filled.append("release date")
            if _filled:
                st.session_state["_mc_pending_game_info"] = _pending
                st.success(f"Filled in {', '.join(_filled)} from Metacritic — check the card "
                           "at the top of the page and adjust if needed.")
            else:
                st.warning("Metacritic didn't return usable title/platform/release-date data "
                           "for this link — fill those in by hand.")
            st.rerun()
    restrict_platform = st.checkbox(
        "Restrict to one platform instead of consolidating all",
        value=False,
        help="Off (default): pulls and merges critic reviews from every platform this game "
             "released on — e.g. a PS5 review AND a separate PC review from the same outlet "
             "both come in, rather than just whichever platform Metacritic's own page shows "
             "by default. On: pick a single platform below.",
    )

    mc_platform_slug = None
    if restrict_platform:
        mc_slug_preview, mc_platform_from_url = metacritic.parse_game_url_or_slug(mc_input) \
            if mc_input.strip() else (None, None)
        if st.button("Look up platforms for this game") and mc_slug_preview:
            try:
                st.session_state["mc_platforms"] = metacritic.fetch_game_platforms(mc_slug_preview)
            except Exception as e:  # noqa: BLE001 - surface a clear error, don't crash the app
                st.error(f"Couldn't look up platforms: {e}")
                st.session_state["mc_platforms"] = []
        platform_options = st.session_state.get("mc_platforms", [])
        if platform_options:
            labels = [f"{p['name']} ({p['review_count']} reviews)" for p in platform_options]
            choice = st.selectbox(
                "Platform", options=range(len(platform_options)), format_func=lambda i: labels[i]
            )
            mc_platform_slug = platform_options[choice]["slug"]
        else:
            st.caption(
                "Click \"Look up platforms\" above to choose one, or leave the checkbox "
                "unchecked to consolidate all platforms."
            )
            if mc_platform_from_url:
                mc_platform_slug = mc_platform_from_url

    if st.button("Fetch reviews from Metacritic") and mc_input.strip():
        mc_slug, mc_platform_from_url = metacritic.parse_game_url_or_slug(mc_input)
        progress = st.progress(0.0, text="Fetching review list from Metacritic...")

        def _mc_progress_cb(done, total, outlet):
            progress.progress(done / max(total, 1), text=f"Fetched {done}/{total}: {outlet}")

        try:
            if restrict_platform:
                effective_platform = mc_platform_slug or mc_platform_from_url
                new_sources = metacritic.build_sources_from_metacritic(
                    mc_slug, platform=effective_platform, progress_cb=_mc_progress_cb,
                    max_workers=parallel_workers,
                )
            else:
                new_sources = metacritic.build_sources_from_metacritic_all_platforms(
                    mc_slug, progress_cb=_mc_progress_cb, max_workers=parallel_workers,
                )
        except Exception as e:  # noqa: BLE001 - surface a clear error, don't crash the app
            st.error(f"Couldn't fetch the review list from Metacritic: {e}")
        else:
            st.session_state["sources"].extend(new_sources)
            failed = [s for s in new_sources if s.error]
            label = (
                f" (platform: {effective_platform})"
                if restrict_platform and effective_platform
                else " (all platforms)"
            )
            st.success(f"Added {len(new_sources)} review(s) from Metacritic{label}.")
            if failed:
                st.warning(
                    f"{len(failed)} of {len(new_sources)} full review texts failed to fetch "
                    "(blocked/JS-only/paywalled) — see the error column in the review table "
                    "below. Use the upload tab for those outlets instead."
                )

with tab_urls:
    urls_text = st.text_area("One review URL per line", height=150)
    if st.button("Fetch these URLs"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        fetched = []
        progress = st.progress(0.0)
        for i, u in enumerate(urls):
            src = fetch_review_url(lambda url: requests.get(url, timeout=20).text,
                                    u, key=f"url_{i}", outlet_hint=u)
            fetched.append(src)
            progress.progress((i + 1) / max(len(urls), 1))
        st.session_state["sources"].extend(fetched)
        failed = [s for s in fetched if s.error]
        if failed:
            st.warning(
                f"{len(failed)} of {len(urls)} URLs failed to fetch (blocked/paywalled/etc). "
                "Use the upload tab for those, or leave them out and note the gap in the report."
            )

with tab_upload:
    st.caption(
        "Zip of .html/.htm files saved from a browser (Ctrl+S / Save Page As) and/or .pdf files "
        "(e.g. printed/saved review pages or scanned clippings), any mix — optionally with a "
        "manifest.csv (columns: filename, outlet, score, date, url). Use this for outlets that "
        "block automated fetching, paywalled reviews, or non-English outlets. A scanned PDF with "
        "no selectable text won't extract — flag it as an error in the review table below. "
        "You don't need to fill in Score by hand: it's auto-detected from the review text "
        "itself when possible (an explicit manifest value always wins), and the Score column "
        "stays editable either way. Reviews in other languages are handled too — the classifier "
        "reads and scores them natively and translates quotes to English for the report; see "
        "the Language column and the run's disclosures for which reviews that applied to."
    )
    zip_file = st.file_uploader("Reviews zip", type=["zip"])
    manifest_file = st.file_uploader("Manifest CSV (optional)", type=["csv"])
    if st.button("Add uploaded reviews") and zip_file:
        new_sources = parse_uploaded_zip(
            zip_file.read(), manifest_file.read() if manifest_file else None
        )
        st.session_state["sources"].extend(new_sources)

# ---------------------------------------------------------------------------
# Review table — fill in/confirm outlet, score, date; drop anything unwanted
# ---------------------------------------------------------------------------
if st.session_state["sources"]:
    st.subheader(f"{len(st.session_state['sources'])} review(s) loaded")
    rows = []
    for s in st.session_state["sources"]:
        on_list = is_on_curated_list(s.outlet, curated_outlets)
        rows.append({
            "key": s.key, "outlet": s.outlet, "platform": s.platform or "",
            "score": s.score, "date": s.date,
            "language": s.language or "— (known after run)",
            "on_curated_list": on_list, "has_text": bool(s.text), "error": s.error or "",
            "include": bool(s.text) and (on_list if restrict_curated else True),
        })
    df = pd.DataFrame(rows)
    edited = st.data_editor(df, num_rows="fixed", disabled=["key", "language", "has_text", "error"],
                             use_container_width=True)

    # Write producer edits (outlet, platform, score, date) back onto the underlying ReviewSource
    # objects — previously only "include"/"on_curated_list" were ever read back out of the
    # edited dataframe, so a hand-corrected outlet name or a hand-filled platform/score typed
    # into this table silently had no effect on the actual run. Keyed by "key", which is
    # disabled/immutable in the editor above, so this lookup is always unambiguous.
    _sources_by_key = {s.key: s for s in st.session_state["sources"]}
    for _, _row in edited.iterrows():
        _src = _sources_by_key.get(_row["key"])
        if _src is None:
            continue
        _src.outlet = _row["outlet"]
        _src.platform = _row["platform"] or None
        _src.score = float(_row["score"]) if pd.notna(_row["score"]) else None
        _src.date = _row["date"] if pd.notna(_row["date"]) and _row["date"] else None

    included_count = int(edited["include"].sum())
    if restrict_curated:
        excluded_off_list = int((~edited["on_curated_list"] & edited["include"]).sum())
        if excluded_off_list:
            st.info(f"{excluded_off_list} included review(s) are not on the curated list snapshot "
                    "— double check outlet name spelling, or uncheck 'restrict to curated list'.")
    if included_count < DEFAULT_MIN_REVIEWS:
        st.warning(f"Only {included_count} reviews included — below the {DEFAULT_MIN_REVIEWS} "
                   "minimum guideline. Results will carry a small-sample caveat.")

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    # A disabled st.button gives no visible reason why — spell it out explicitly rather than
    # leaving the producer to guess which precondition they're missing.
    disabled_reasons = []
    if not game_title:
        disabled_reasons.append('enter a **Game title** above (in the card at the top of the page)')
    if not categories:
        disabled_reasons.append("select or add at least one **category** above")
    if included_count == 0:
        disabled_reasons.append('check **include** for at least one review in the table above')

    if disabled_reasons:
        st.warning("Run metareview is disabled until you " + "; and ".join(disabled_reasons) + ".")

    if st.button("Run metareview", type="primary", disabled=bool(disabled_reasons)):
        if not categories:
            st.error("Select at least one category (standard or custom) before running.")
            st.stop()
        if client is None:
            st.error("No LLM credentials configured — set AWS_BEDROCK_* or ANTHROPIC_API_KEY in secrets.")
            st.stop()

        included_keys = set(edited.loc[edited["include"], "key"])
        active_sources = [s for s in st.session_state["sources"] if s.key in included_keys]

        data = {}  # category -> {key: (value, quote)}
        all_candidates = []
        failed_keys = set()
        failed_sources = []  # (outlet, error) — one bad review shouldn't sink the whole batch
        non_english = {}  # outlet -> language, for the translated-quotes disclosure below
        llm_filled_scores = 0  # scores backfilled from the model, not the manifest/regex/hand-edit
        progress = st.progress(0.0, text="Classifying reviews...")

        def _classify_progress_cb(done, total):
            progress.progress(done / max(total, 1), text=f"Classified {done}/{total}")

        # Classified concurrently (bounded by the "Parallel requests" sidebar slider) — see
        # classify.classify_reviews()'s docstring. Its result list is guaranteed to line up
        # positionally with active_sources regardless of which review's API call actually
        # finished first, so the zip() below is safe.
        classify_results = classify.classify_reviews(
            client, active_sources, categories, model=model,
            max_workers=parallel_workers, progress_cb=_classify_progress_cb,
        )
        for s, result in zip(active_sources, classify_results):
            if result is None or "error" in result:
                err = result["error"] if result else "no result returned"
                failed_keys.add(s.key)
                failed_sources.append((s.outlet, err))
                continue
            for c in result["classifications"]:
                data.setdefault(c["category"], {})[s.key] = (c["value"], c["quote"])
            all_candidates.extend(result.get("candidate_emergent_topics", []))

            # Score: the table's existing value (manifest, upload-time regex extraction, or a
            # producer's hand-edit) always wins — the LLM only fills in what's still None.
            if s.score is None and result.get("stated_score") is not None:
                s.score = result["stated_score"]
                llm_filled_scores += 1

            # Language: every classification quote above was already translated to English per
            # the classification prompt — this just records which reviews needed it.
            s.language = result.get("review_language") or "Unknown"
            if s.language.lower() not in ("english", "unknown"):
                non_english[s.outlet] = s.language

        if failed_sources:
            st.warning(
                f"{len(failed_sources)} of {len(active_sources)} review(s) failed to classify "
                "and were excluded from scoring rather than aborting the whole run — re-run just "
                "those, or leave them out and note the gap in the report: "
                + ", ".join(f"{outlet} ({err})" for outlet, err in failed_sources)
            )
        # Drop failed reviews before they reach scoring — otherwise they'd still count toward
        # the review-count denominator in compute_weighted_scores with zero contributed
        # sentiment, silently diluting every category's weighted score.
        active_sources = [s for s in active_sources if s.key not in failed_keys]
        if not active_sources:
            st.error("All included reviews failed to classify — nothing left to score. "
                      "See the warning above for per-review errors, fix or drop those, and re-run.")
            st.stop()

        emergent = classify.detect_emergent_categories(client, all_candidates, len(active_sources),
                                                         model=model)
        full_categories = categories + [c for c in emergent if c not in categories]

        emergent_failed = []
        if emergent:
            # Emergent categories are only just NAMED at this point — no review has actually
            # been asked about them yet (they weren't known to exist during the classification
            # pass above). Without this second pass, every emergent category would carry real
            # names into the matrix with zero classifications behind them — not "no reviews
            # mentioned this" but "no review was ever asked" — which renders as a flat,
            # misleading 0 across every stat instead of real data. See
            # classify.classify_reviews_for_categories()'s docstring for more.
            progress2 = st.progress(0.0, text="Scoring reviews against emergent categories...")

            def _emergent_progress_cb(done, total):
                progress2.progress(done / max(total, 1),
                                    text=f"Scored {done}/{total} against emergent categories")

            emergent_data, emergent_failed = classify.classify_reviews_for_categories(
                client, active_sources, emergent, model=model, max_workers=parallel_workers,
                progress_cb=_emergent_progress_cb
            )
            for cat, entries in emergent_data.items():
                data.setdefault(cat, {}).update(entries)
            if emergent_failed:
                st.warning(
                    f"{len(emergent_failed)} of {len(active_sources)} review(s) failed to score "
                    "against the emergent categories specifically (their standard-category "
                    "scores from the main pass above are unaffected): "
                    + ", ".join(f"{outlet} ({err})" for outlet, err in emergent_failed)
                )

        review_dicts = [{"key": s.key, "outlet": s.outlet, "score": s.score, "date": s.date,
                          "platform": s.platform} for s in active_sources]
        scores = compute_weighted_scores(review_dicts, full_categories, data, INCLUSION_THRESHOLD)

        # None (not 0.00) when no review in the batch has a stated score — a real, legitimate
        # outcome (not every outlet publishes a numeric score), and 0.00 would misleadingly
        # read as a universally-panned game rather than "no score data available."
        scored_values = [r["score"] for r in review_dicts if r["score"] is not None]
        avg_score = round(sum(scored_values) / len(scored_values), 2) if scored_values else None

        st.session_state["result"] = {
            "reviews": review_dicts, "categories": full_categories, "data": data,
            "scores": scores, "emergent": emergent, "non_english": non_english,
            "llm_filled_scores": llm_filled_scores, "avg_score": avg_score,
        }
        st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    res = st.session_state["result"]
    st.divider()
    st.header("Results")

    reportable = {k: v for k, v in res["scores"].items() if v["meets_threshold"]}
    below = {k: v for k, v in res["scores"].items() if not v["meets_threshold"]}

    chart_df = pd.DataFrame([
        {"category": k, "weighted": v["weighted"]} for k, v in reportable.items()
    ]).sort_values("weighted")
    if not chart_df.empty:
        fig = px.bar(chart_df, x="weighted", y="category", orientation="h",
                     title="Opinion graph (weighted)", color="weighted",
                     color_continuous_scale=["#CC2244", "#1A2A4A", "#00BB66"], range_color=[-1, 1])
        fig.update_layout(paper_bgcolor="#0D1B2A", plot_bgcolor="#0D1B2A",
                           font_color="#E0E0E0")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame([
        {"category": k, "weighted": round(v["weighted"], 2), "positive": v["positive"],
         "mixed": v["mixed"], "negative": v["negative"],
         "mention_rate": f"{v['mention_rate']:.0%}"}
        for k, v in res["scores"].items()
    ]), use_container_width=True)

    if below:
        st.caption(f"Below the {INCLUSION_THRESHOLD:.0%} reporting threshold this run: "
                   + ", ".join(below.keys()))
    if res["emergent"]:
        st.success(f"Emergent categories detected: {', '.join(res['emergent'])}")
    if res.get("avg_score") is None:
        st.warning(
            "No review in this batch had a detectable numeric score — Average Score will show "
            "as N/A rather than 0.00. This doesn't affect sentiment scoring (which never uses "
            "the numeric score), but double check: are these outlets genuinely score-less "
            "prose reviews, or did score extraction miss something (e.g. a score shown as an "
            "image/badge in the source rather than as text)? Scores are still hand-editable in "
            "the review table above if you want to fill any in manually and re-run."
        )

    wb = build_matrix_workbook(game_title, res["reviews"], res["categories"], res["data"])
    xlsx_buf = BytesIO()
    wb.save(xlsx_buf)

    platform_breakdown = compute_platform_averages(res["reviews"])
    if len(platform_breakdown) >= 2:
        st.caption("Platform breakdown (will also appear in the report's Review Averages):")
        st.dataframe(pd.DataFrame([
            {"platform": p, "review_count": s["review_count"],
             "average_score": s["average_score"] if s["average_score"] is not None else "N/A"}
            for p, s in platform_breakdown.items()
        ]), use_container_width=True)

    non_english = res.get("non_english", {})
    disclosures = [
        f"{len(res['reviews'])} reviews used (target was {target_reviews}; "
        f"{DEFAULT_MIN_REVIEWS} minimum per Sega's existing guidance).",
        "Sourced via direct outlet fetch and/or producer-uploaded saved pages — see run inputs.",
        f"Categories below the {INCLUSION_THRESHOLD:.0%} mention threshold were excluded from "
        f"scoring: {', '.join(below.keys()) if below else 'none'}.",
        "Scores were taken from a manifest.csv, a producer's hand-edit in the review table, an "
        "auto-detected numeric score found in the review text itself (e.g. '8.5/10', '90%'), or "
        f"— for {res.get('llm_filled_scores', 0)} review(s) this run — the classifying model's "
        "own reading of a score stated in the text that the automatic pass missed. A review "
        "with no explicit numeric score anywhere has a blank Score cell; that's expected, not "
        "an error, and doesn't exclude it from sentiment scoring.",
        (
            f"{len(non_english)} review(s) were not in English ({', '.join(sorted(set(non_english.values())))}) "
            f"— {', '.join(f'{o} ({l})' for o, l in non_english.items())}. Their quotes in this "
            "report were translated to English by the classifying model as part of the same "
            "call that scored them — verify translations before treating them as exact, the "
            "same as any other AI-drafted content in this run."
            if non_english else
            "All included reviews were detected as English-language — no translation was needed."
        ),
        "Narrative sections (Executive Summary, category call-outs, recommendations) are "
        "AI-drafted from the matrix data — a producer should review every quote and "
        "conclusion before this is treated as final.",
    ]
    if res.get("emergent"):
        disclosures.append(
            f"{len(res['emergent'])} emergent categor{'y' if len(res['emergent']) == 1 else 'ies'} "
            f"detected from recurring themes reviewers raised that weren't on the standard/"
            f"custom category list: {', '.join(res['emergent'])}. Every included review was "
            "classified a second time against just these categories, so they carry real scored "
            "data in the matrix (not just a name)."
        )

    with st.spinner("Drafting narrative sections..."):
        drafted = narrative.draft_narrative(
            client, game_title, platforms, res["avg_score"], len(res["reviews"]),
            res["scores"], res["data"], res["reviews"], model=model,
            include_recommendations=include_recommendations,
            fan_reaction_notes=fan_reaction_notes,
        )

    docx_bytes = build_report_docx(
        game_title=game_title,
        methodology_note=(
            f"Data points collected from {len(res['reviews'])} press reviews. Sentiment scored "
            "per Sega's existing critique rubric (see Methodology tab in the accompanying skill)."
        ),
        narrative=drafted,
        review_count=len(res["reviews"]), average_score=res["avg_score"],
        disclosures=disclosures,
        platform_breakdown=platform_breakdown,
    )

    with st.expander("Drafted narrative (preview before download)"):
        st.write(drafted.get("executive_summary", ""))
        for co in drafted.get("category_callouts", []):
            st.markdown(f"**{co['category']} [{co['label']}]**")
            if co.get("label_caveat"):
                st.caption(f"*{co['label_caveat']}*")
            st.write(co.get("synthesis", ""))
        if drafted.get("fan_reaction"):
            st.markdown("**Impression of Fan Reaction**")
            for para in drafted["fan_reaction"]:
                st.write(para)

    dl1, dl2 = st.columns(2)
    dl1.download_button("Download matrix (.xlsx)", xlsx_buf.getvalue(),
                         file_name=f"{game_title or 'metareview'}_matrix.xlsx")
    dl2.download_button("Download report draft (.docx)", docx_bytes,
                         file_name=f"{game_title or 'metareview'}_report.docx")

    with st.expander("Disclosures included in this run"):
        for d in disclosures:
            st.write(f"- {d}")
