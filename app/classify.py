"""
Claude API calls: per-review classification against the rubric, cross-review emergent-category
detection, and a shared batch-classification helper used for both the primary category pass and
the follow-up pass that actually scores reviews against whatever emergent categories get found
(see classify_reviews_for_categories()'s docstring — this second pass is required, not optional;
skipping it is what causes emergent categories to show up in the matrix with real names but
every stat flatly at 0).

Client resolution matches the pattern used across Sega's other internal Streamlit tools (see
e.g. narrative_qa.py): AWS Bedrock first (separate credential set from SES), falling back to
a direct Anthropic API key if Bedrock isn't configured. All credentials come from st.secrets —
see .streamlit/secrets.toml.example.

The matrix/report-building plumbing is exercised without any live credentials in
tests/test_pipeline.py by calling classify_review/detect_emergent_categories with a hand-built
fake `client`; get_client() itself needs real secrets and is not covered by that suite.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import streamlit as st

try:
    from anthropic import Anthropic, AnthropicBedrock
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .rubric import CLASSIFICATION_SYSTEM_PROMPT, INCLUSION_THRESHOLD

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"  # Bedrock model ID; see get_client()

# Bounded concurrency for classify_reviews() below: these calls are network-I/O-bound (waiting
# on the Anthropic/Bedrock API), not CPU-bound, so a ThreadPoolExecutor is effective despite the
# GIL — threads spend nearly all their time blocked on I/O, not contending for the interpreter
# lock. Deliberately NOT unbounded ("one thread per review"): a large batch firing dozens of
# simultaneous requests risks tripping Anthropic/Bedrock's per-minute rate limit. 5 is a
# reasonable default; streamlit_app.py exposes this as a sidebar slider so a producer can tune it.
DEFAULT_MAX_WORKERS = 5

def _build_classify_tool(categories: List[str]) -> dict:
    """
    Builds the classification tool schema for THIS call, constraining the "category" field to
    an enum of the exact given category strings.

    Without this, nothing stops the model from paraphrasing a category name back slightly
    differently than it was given. Harmless for a short name like "Combat" that's trivial to
    echo verbatim — but a real, silent data-loss bug for a long compound name like "Missing
    Content from Prior Versions (FES/Portable/The Answer)": a paraphrased return value doesn't
    match the exact key used to build the matrix (`data.setdefault(c["category"], ...)`), so
    that review's classification for that category just lands in an untracked dict key that
    never reaches compute_weighted_scores or the report — confirmed against a real 35-review
    run where every custom category came back with zero classifications while every short
    default category didn't. The enum constraint (plus the defense-in-depth filter in
    _normalize_classifications) closes this off structurally rather than hoping the model
    happens to echo strings back exactly.
    """
    return {
        "name": "submit_classification",
        "description": "Submit the category-by-category classification for this single review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": categories},
                            "value": {"type": "integer", "enum": [-1, 0, 1]},
                            "quote": {"type": "string"},
                        },
                        "required": ["category", "value", "quote"],
                    },
                },
                "candidate_emergent_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recurring topics with a clear opinion, not on the given category list.",
                },
                "review_language": {
                    "type": "string",
                    "description": "Primary language of the review text, as a plain English "
                                    "language name (e.g. 'English', 'French', 'Japanese').",
                },
                "stated_score": {
                    "type": ["number", "null"],
                    "description": "The review's own explicit numeric score, normalized to a 0-100 "
                                    "scale. null if the review states no numeric score — never "
                                    "infer one from tone.",
                },
                "stated_score_raw": {
                    "type": ["string", "null"],
                    "description": "The score exactly as written in the review (e.g. '8.5/10'). "
                                    "null if stated_score is null.",
                },
            },
            "required": ["classifications", "candidate_emergent_topics", "review_language",
                          "stated_score", "stated_score_raw"],
        },
    }

CLUSTER_TOOL = {
    "name": "submit_clusters",
    "description": "Submit clustered emergent-topic candidates with per-cluster mention counts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short category name for this cluster."},
                        "mention_count": {"type": "integer"},
                    },
                    "required": ["label", "mention_count"],
                },
            },
        },
        "required": ["clusters"],
    },
}


@st.cache_resource
def get_client():
    """Bedrock first (uses AWS_BEDROCK_* secrets, separate from the SES creds auth.py uses),
    falling back to a direct Anthropic key. Returns None if neither is configured — callers
    should check and surface a clear error rather than let requests fail deep in the call."""
    if not HAS_ANTHROPIC:
        return None
    try:
        return AnthropicBedrock(
            aws_access_key=st.secrets["AWS_BEDROCK_ACCESS_KEY_ID"],
            aws_secret_key=st.secrets["AWS_BEDROCK_SECRET_ACCESS_KEY"],
            aws_region=st.secrets.get("AWS_BEDROCK_REGION", "us-east-1"),
        )
    except Exception:  # noqa: BLE001 - missing/invalid secrets, try the next option
        pass
    try:
        return Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except Exception:  # noqa: BLE001
        return None


def _normalize_classifications(raw, valid_categories: Optional[List[str]] = None) -> List[dict]:
    """
    Defensively coerces the model's raw `classifications` output into the guaranteed shape
    [{"category": str, "value": -1|0|1, "quote": str}, ...].

    Forced tool-use (tool_choice) is reliable in the overwhelming common case, but it's a model
    behavior, not a hard server-side contract — it has been observed to occasionally return
    `classifications` as a {category: value} mapping instead of an array of objects, or to
    include a bare category-name string with no value/quote attached. Either shape used to
    reach `data.setdefault(c["category"], ...)` in streamlit_app.py unchanged and crash the
    *entire* run with "string indices must be integers, not 'str'" on whichever review happened
    to trigger it — after however many reviews' worth of paid LLM calls had already completed.
    Coercing/dropping bad entries here means a single odd response degrades that one review's
    data instead of losing the whole batch.

    valid_categories, when given, drops any classification whose category isn't an exact match
    for one of the categories this call was actually asked about — defense-in-depth alongside
    the enum constraint in _build_classify_tool, in case a backend doesn't enforce the schema's
    enum as strictly as the direct Anthropic API does.
    """
    if isinstance(raw, dict):
        # Model returned a category -> value/quote mapping instead of an array of objects.
        raw = [
            {"category": k, **(v if isinstance(v, dict) else {"value": v})}
            for k, v in raw.items()
        ]
    if not isinstance(raw, list):
        return []

    valid_set = set(valid_categories) if valid_categories is not None else None
    normalized = []
    for item in raw:
        if not isinstance(item, dict):
            continue  # e.g. a bare category-name string — no value to score it with, drop it
        category, value = item.get("category"), item.get("value")
        if not isinstance(category, str) or not category.strip() or value not in (-1, 0, 1):
            continue
        if valid_set is not None and category not in valid_set:
            continue
        normalized.append({"category": category, "value": value, "quote": item.get("quote") or ""})
    return normalized


def _normalize_review_language(value) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Unknown"  # model omitted/malformed this field — don't let it crash the caller


def _normalize_stated_score(value) -> Optional[float]:
    """0-100 scale, matching the rest of the app's convention. Rejects bools (True/False are
    technically `int` subclasses in Python and would otherwise sneak through as 1/0) and
    anything outside the valid range rather than trusting the model's arithmetic blindly."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if 0 <= v <= 100:
            return round(v, 1)
    return None


def classify_review(client, review_text: str, categories: List[str],
                     model: str = DEFAULT_MODEL) -> dict:
    """
    Returns {"classifications": [...], "candidate_emergent_topics": [...], "review_language":
    str, "stated_score": float|None, "stated_score_raw": str|None} — always in this normalized
    shape regardless of what the model actually returned (see _normalize_classifications).

    review_language/stated_score* let a caller backfill a review's Score column and flag which
    reviews needed quote translation, without a separate API call — see streamlit_app.py.
    """
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=CLASSIFICATION_SYSTEM_PROMPT,
        tools=[_build_classify_tool(categories)],
        tool_choice={"type": "tool", "name": "submit_classification"},
        messages=[{
            "role": "user",
            "content": (
                f"Category list for this game:\n{', '.join(categories)}\n\n"
                f"Review text:\n{review_text}"
            ),
        }],
    )
    for block in msg.content:
        if block.type == "tool_use" and block.name == "submit_classification":
            raw = block.input or {}
            stated_score_raw = raw.get("stated_score_raw")
            return {
                "classifications": _normalize_classifications(
                    raw.get("classifications", []), valid_categories=categories
                ),
                "candidate_emergent_topics": [
                    t for t in raw.get("candidate_emergent_topics", []) if isinstance(t, str) and t.strip()
                ],
                "review_language": _normalize_review_language(raw.get("review_language")),
                "stated_score": _normalize_stated_score(raw.get("stated_score")),
                "stated_score_raw": stated_score_raw if isinstance(stated_score_raw, str) else None,
            }
    raise RuntimeError("Model did not return a submit_classification tool call")


def classify_reviews(client, sources, categories: List[str], model: str = DEFAULT_MODEL,
                      max_workers: int = DEFAULT_MAX_WORKERS,
                      progress_cb: Optional[Callable[[int, int], None]] = None) -> List[dict]:
    """
    Classifies every item in `sources` against `categories` concurrently (bounded by
    max_workers workers), returning one result dict per source in the SAME ORDER as `sources` —
    not completion order. Order matters here: callers (e.g. metacritic._make_key, and the main
    classification loop in streamlit_app.py) key results back to sources by position, so a
    caller must be able to zip(sources, classify_reviews(...)) and get the right pairing
    regardless of which review's API call actually finished first.

    Each entry is either classify_review()'s normal result dict, or {"error": str} if that
    specific review's classification call raised — isolated per-review, same principle used
    throughout this app: one bad review's failure doesn't sink the whole batch.

    Concurrency is via ThreadPoolExecutor rather than multiprocessing or plain asyncio: these
    calls are network-I/O-bound (waiting on the Anthropic/Bedrock API), not CPU-bound, so threads
    spend nearly all their time blocked rather than contending for the GIL — see DEFAULT_MAX_WORKERS
    above for why concurrency is bounded rather than "one thread per review."

    progress_cb(done, total), if given, is called from the calling (main) thread as each result
    completes — as_completed() is iterated on the caller's thread even though the futures
    themselves ran on worker threads, so this is safe to wire directly to Streamlit progress
    widgets, which are not themselves safe to call from inside a worker thread.
    """
    results: List[Optional[dict]] = [None] * len(sources)
    done_count = 0

    def _run(i, s):
        return i, classify_review(client, s.text, categories, model=model)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(_run, i, s): i for i, s in enumerate(sources)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, result = future.result()
            except Exception as e:  # noqa: BLE001 - isolate this review, keep the batch going
                result = {"error": str(e)}
            results[i] = result
            done_count += 1
            if progress_cb:
                progress_cb(done_count, len(sources))
    return results


def classify_reviews_for_categories(client, sources, categories: List[str],
                                     model: str = DEFAULT_MODEL,
                                     max_workers: int = DEFAULT_MAX_WORKERS,
                                     progress_cb: Optional[Callable[[int, int], None]] = None):
    """
    Classifies every item in `sources` against `categories` (concurrently, via classify_reviews()
    above) and merges the results into a single {category: {review_key: (value, quote)}} dict —
    the same shape streamlit_app.py's main classification loop used to build by hand around
    classify_review(), factored out here so it can also drive the follow-up pass that scores
    reviews against EMERGENT categories once detect_emergent_categories() has named them.

    That follow-up pass is necessary, not optional: emergent categories aren't known until every
    review has already been classified once (detect_emergent_categories clusters candidate
    topics pooled across all of them), so there is no way to score reviews against an emergent
    category in the same pass that discovers it. Skipping this second pass — which the app did
    for a while — silently leaves every emergent category with zero classifications: not "no
    reviews mentioned this" but "no review was ever actually asked about this," which renders as
    a misleading flat 0 (weighted score, mention_rate, positive/mixed/negative all 0) in the
    matrix and report, indistinguishable from genuine unanimous silence on the topic.

    `sources` needs only a `.key`, `.outlet`, and `.text` attribute per item — a real
    ReviewSource, or anything duck-typed the same way for testing.

    Returns (data, failed) — failed is [(outlet, error_message), ...] for any review whose
    classification call raised, isolated per-review (same principle as the main classification
    loop: one bad review's failure here doesn't sink the whole batch, and doesn't affect that
    review's already-scored standard-category classifications either).

    progress_cb(done, total), if given, is called after each review completes — same shape as
    the progress callbacks already used elsewhere in this app (e.g. metacritic.py). max_workers=1
    makes this fully sequential/deterministic — used by a couple of tests whose fakes assume a
    fixed call order; real callers should leave it at the default.
    """
    results = classify_reviews(client, sources, categories, model=model,
                                max_workers=max_workers, progress_cb=progress_cb)
    data = {}
    failed = []
    for s, result in zip(sources, results):
        if result is None:
            continue
        if "error" in result:
            failed.append((s.outlet, result["error"]))
            continue
        for c in result["classifications"]:
            data.setdefault(c["category"], {})[s.key] = (c["value"], c["quote"])
    return data, failed


def detect_emergent_categories(client, all_candidates: List[str], num_reviews: int,
                                model: str = DEFAULT_MODEL,
                                threshold: float = INCLUSION_THRESHOLD) -> List[str]:
    """
    Clusters free-text candidate topics (gathered across all per-review classify_review calls)
    into distinct themes and returns those mentioned by enough reviews to clear the inclusion
    threshold. Returns [] if there aren't enough candidates to bother clustering.
    """
    if not all_candidates:
        return []
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=(
            "You are clustering short topic phrases pulled from many separate game reviews "
            "into a small number of distinct recurring themes. Merge near-duplicates (e.g. "
            "'slow start' and 'sluggish opening pacing' are the same cluster). Count how many "
            "of the input phrases fall into each cluster — that's a proxy for how many reviews "
            "raised it, so count generously-but-honestly, don't invent phrases that weren't given."
        ),
        tools=[CLUSTER_TOOL],
        tool_choice={"type": "tool", "name": "submit_clusters"},
        messages=[{"role": "user", "content": "\n".join(f"- {c}" for c in all_candidates)}],
    )
    for block in msg.content:
        if block.type == "tool_use" and block.name == "submit_clusters":
            clusters = block.input.get("clusters", []) if block.input else []
            if not isinstance(clusters, list):
                return []
            out = []
            for c in clusters:
                if not isinstance(c, dict):
                    continue  # same defensive stance as _normalize_classifications
                label, count = c.get("label"), c.get("mention_count")
                if isinstance(label, str) and label.strip() and isinstance(count, (int, float)):
                    if num_reviews and (count / num_reviews) >= threshold:
                        out.append(label)
            return out
    return []
