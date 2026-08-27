"""
The narrative-drafting agent — the piece that was previously just a placeholder string in
streamlit_app.py ("[Draft — replace with a producer-reviewed summary.]"). This module actually
calls Claude to write the Executive Summary, Press Reactions synthesis, category call-outs, and
PD Recommendations, in the house style derived from two reference reports across two different
genres (Persona 3: Reload — RPG, Puyo Puyo Tetris — puzzle/party game).

What's genre-INVARIANT across those two reports (and so belongs in the system prompt, not
per-run data):
  - Methodology paragraph phrasing/structure (critique definition, 25% threshold, press-only
    scope caveat).
  - Executive Summary is 3-5 sentences: overall reception, then the 2-3 strongest points, then
    the 1-2 weakest, in that order.
  - Category call-outs use a bracketed severity label in the heading (see
    matrix.suggest_sentiment_label) and are written for EVERY category in the matrix, not just a
    hand-picked "most significant" subset — a category with only a handful of mentions still
    gets at least a short paragraph (with a caveat that the sample is thin), rather than being
    silently left out of the report entirely. Depth scales with how much real data backs a
    category; coverage doesn't.
  - Each call-out is a short synthesis paragraph followed by 2-5 directly attributed quotes
    (fewer, or none, for a thin category with little to quote).
  - A producer may append a one-line caveat to a label when the sample behind it is thin (the
    "Richard's Note" pattern) — the model should do the same rather than presenting a
    thin-sample label with false confidence.
  - PD Recommendations are a list of short, specifically-titled, actionable sub-sections tied
    back to a category or theme — not generic advice.

What's genre-VARIANT (comes from the actual run data, not the prompt):
  - The category list itself.
  - Whether a "Review Averages" breakdown by storefront/platform makes sense (P3R didn't split
    this; Puyo Puyo Tetris did, by console).
  - Whether an "Impression of Fan Reaction" section appears at all — it's drafted only when the
    producer supplies raw fan/user notes for this run (streamlit_app.py's optional text area,
    threaded through as draft_narrative's fan_reaction_notes param), and is always explicitly
    impressionistic rather than a scored/threshold-based finding like the press sections above.
  - Whether a "Review" sub-section under Review and Recommendations exists at all — Puyo Puyo
    Tetris explicitly omitted it ("first entry in this franchise in the west").
"""
from typing import List, Optional

def _build_category_callouts_property(categories: List[str]) -> dict:
    """
    `category` is constrained to an enum of the exact given category names — same defensive
    reasoning as classify.py's _build_classify_tool: without this, nothing stops the model from
    paraphrasing a category name back slightly differently, which would make the completeness
    check in draft_narrative() below (matching returned categories against the given set) treat
    a real write-up as "missing" and inject a redundant fallback paragraph next to it.
    """
    return {
        "type": "array",
        "description": (
            "One write-up per category — EVERY category listed below, no exceptions, even one "
            "mentioned by only a handful of reviews. A well-supported category gets a fuller "
            "synthesis (2-4 sentences, several quotes); a thin one can be as short as a single "
            "sentence with 0-2 quotes and a label_caveat noting the small sample. Completeness "
            "matters here, not depth — never drop a category from this array."
        ),
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": categories},
                "label": {
                    "type": "string",
                    "description": "The severity label for the heading, e.g. 'Widely "
                                   "Praised' or 'Extremely Controversial'. Start from the "
                                   "suggested_label given to you; override with your own "
                                   "if the data clearly warrants it.",
                },
                "label_caveat": {
                    "type": "string",
                    "description": "One-line caveat if this category's sample is thin "
                                   "enough that the label deserves a grain of salt "
                                   "(the 'Richard's Note' pattern) — empty string if not needed.",
                },
                "synthesis": {"type": "string", "description": "1-3 sentence synthesis paragraph "
                                                                 "(1 sentence is fine for a thin category)."},
                "quotes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "outlet": {"type": "string"},
                        },
                        "required": ["text", "outlet"],
                    },
                    "description": "2-5 directly attributed quotes pulled verbatim from the "
                                   "quotes you were given — never invent a quote. Fewer (down "
                                   "to 0) is fine for a category with little to quote from.",
                },
            },
            "required": ["category", "label", "synthesis", "quotes"],
        },
    }

_FAN_REACTION_PROPERTY = {
    "type": "array",
    "items": {"type": "string"},
    "description": "2-4 short paragraphs synthesizing the raw fan/user reaction notes you were "
                    "given, in the style of an 'Impression of Fan Reaction' section. This is "
                    "explicitly impressionistic, not a scored/threshold-based finding like the "
                    "press-review sections above — say so, and draw only from the notes you "
                    "were actually given. Never invent fan sentiment that wasn't provided.",
}

_RECOMMENDATIONS_PROPERTY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "heading": {"type": "string", "description": "Short, specific, e.g. 'Keeping the Story Light'."},
            "text": {"type": "string"},
        },
        "required": ["heading", "text"],
    },
    "description": "PD Recommendations for the next title — specific and actionable, "
                    "each tied to a category or cross-cutting theme, not generic advice.",
}


def _build_narrative_tool(categories: List[str], include_recommendations: bool,
                           include_fan_reaction: bool = False) -> dict:
    """
    categories: every category name that must get a call-out (see
    _build_category_callouts_property) — pass list(category_scores.keys()) from the run's full
    matrix, not a pre-filtered subset.

    include_recommendations=False omits the "recommendations" property from the schema
    entirely, rather than asking the model to produce one and then discarding it — a producer
    who wants PD Recommendations left out of a given run (e.g. it's handled elsewhere in their
    process for this title) shouldn't have the model spend output budget drafting something
    that's just going to be thrown away.

    include_fan_reaction (default False, unlike include_recommendations) adds the "fan_reaction"
    property — only meaningful when the producer actually supplied raw fan/user notes for this
    run (see draft_narrative's fan_reaction_notes param), so the caller decides this from
    whether that text is present, not a standing sidebar toggle.
    """
    properties = {
        "executive_summary": {
            "type": "string",
            "description": "3-5 sentences: overall reception, then strongest points, then weakest.",
        },
        "press_reactions_synthesis": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 short paragraphs synthesizing cross-cutting press reaction "
                            "themes (things that show up across multiple categories), in the "
                            "style of a 'Press Reactions' section intro.",
        },
        "category_callouts": _build_category_callouts_property(categories),
    }
    required = ["executive_summary", "press_reactions_synthesis", "category_callouts"]
    if include_recommendations:
        properties["recommendations"] = _RECOMMENDATIONS_PROPERTY
        required.append("recommendations")
    if include_fan_reaction:
        properties["fan_reaction"] = _FAN_REACTION_PROPERTY
        required.append("fan_reaction")

    return {
        "name": "submit_narrative",
        "description": "Submit the drafted narrative sections for this metareview report.",
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }

def _build_system_prompt(include_recommendations: bool, include_fan_reaction: bool = False) -> str:
    recommendations_section = (
        """

PD Recommendations: specific, actionable sub-sections for the next title in this series or genre, \
each with a short pointed heading (e.g. "Keeping the Story Light", not "Improve the Story"). Tie \
every recommendation back to a specific category or cross-cutting theme from this review — no \
generic advice that could apply to any game."""
        if include_recommendations else
        """

This run has been configured with PD Recommendations intentionally left out of scope — do not \
submit a "recommendations" field at all (it isn't part of the schema you were given for this call)."""
    )
    fan_reaction_section = (
        """

Impression of Fan Reaction: 2-4 short paragraphs synthesizing the raw fan/user notes given to \
you below (forum threads, storefront user reviews, survey responses, etc. — supplied separately \
from, and NOT part of, the scored press-review data above). This section is explicitly \
impressionistic, not a scored/threshold-based finding — say so plainly rather than presenting it \
with the same confidence as the press-review sections, and never invent fan sentiment beyond \
what the notes actually say."""
        if include_fan_reaction else ""
    )
    reportable_requirement = (
        "category_callouts must have exactly one entry per category given to you (see the "
        "category list in the schema's enum) and recommendations must NOT be an empty array"
        if include_recommendations else
        "category_callouts must have exactly one entry per category given to you (see the "
        "category list in the schema's enum)"
    )
    return f"""You are drafting the narrative sections of a Sega metareview report. \
A metareview aggregates press-critic sentiment into a category-by-category matrix, then a report \
synthesizes that matrix into prose for a Post-Launch Review meeting. You are given the finished \
matrix (per-category weighted scores, positive/mixed/negative counts, a deterministic suggested \
severity label per category, and a pool of supporting quotes) — your job is ONLY to write the \
narrative around it, matching Sega's existing house style.

HOUSE STYLE (derived from prior reports — follow this structure and tone):

Executive Summary: 3-5 sentences. State overall reception first (in relation to the average \
score), then name the 2-3 strongest categories, then the 1-2 weakest. Plain, factual tone — this \
is a business document read by a Director/VP, not marketing copy.

Press Reactions synthesis: 2-4 short paragraphs describing THEMES that cut across multiple \
categories — e.g. "reviewers praised the breadth of content across every mode," or "criticism of \
X was often really about Y." This is not a list of every category; it's the connective tissue \
between them.

Category call-outs: write ONE entry for EVERY category in the matrix — do not skip any, and do \
not limit yourself to a "most significant" subset. Depth should scale with how much real data \
backs a category, not whether it makes the cut at all:
- A well-supported category (comfortably above the reporting threshold, several quotes \
available) gets the full treatment: 2-4 sentence synthesis, 2-5 quotes.
- A thin category (barely mentioned, few or no quotes to draw on) still gets its own entry — as \
short as one sentence, with 0-2 quotes if that's all there is, plus a label_caveat noting the \
small sample. A short, honest entry beats no entry at all; a producer scanning the report should \
never have to wonder whether a category was overlooked versus genuinely silent.

For each call-out:
- Use the given suggested_label for the heading (e.g. "Widely Praised", "Extremely \
Controversial") unless the data clearly warrants a different one — these labels come from a \
fixed vocabulary: Universally/Widely/Generally Praised or Panned, Somewhat Controversial \
(Leaning Positive/Negative), Controversial (Leaning Positive/Negative), Extremely Controversial.
- If the category's total mention count is small relative to the review pool (a thin sample), \
add a one-line label_caveat noting that — do not present a low-confidence label with false \
confidence. This mirrors a real producer's own practice of flagging exactly this.
- Write a short synthesis (1-3 sentences; 1 is fine for a thin category), then attach up to 5 \
quotes PULLED VERBATIM from the quote pool you were given, each with its outlet attribution — \
0 quotes is fine if the category genuinely has little or nothing to draw on. Never invent or \
paraphrase a quote as if it were verbatim — if you want to paraphrase, do it in the synthesis \
text, not inside a quote field.{recommendations_section}{fan_reaction_section}

Do not use marketing language, exclamation points, or hedge everything into mush. State \
conclusions plainly. This document's reader already trusts the underlying data (the matrix) — \
your job is to make it readable, not to sell it.

You are given category stats for every category in this run's matrix, thin ones included — {reportable_requirement}. \
If you are running low on output space, favor shorter call-outs across ALL categories over full-length \
call-outs for only some — completeness matters more here than depth on any single one."""


def build_narrative_user_prompt(game_title: str, platforms: str, avg_score: float,
                                  review_count: int, category_scores: dict, data: dict,
                                  reviews: list, fan_reaction_notes: Optional[str] = None) -> str:
    """category_scores: output of matrix.compute_weighted_scores (has suggested_label).
    data: {category: {review_key: (value, quote)}} — the quote pool.
    reviews: list of {key, outlet, score, date} — for resolving outlet names.
    fan_reaction_notes: raw producer-supplied fan/user commentary, appended verbatim as its own
    block at the end when given — see draft_narrative's fan_reaction_notes param."""
    outlet_by_key = {r["key"]: r["outlet"] for r in reviews}

    avg_score_text = f"{avg_score:.1f}" if avg_score is not None else "N/A (no review in this batch stated a numeric score)"
    lines = [
        f"Game: {game_title}", f"Platform(s): {platforms}",
        f"Reviews: {review_count}, average score {avg_score_text}", "",
        "Category stats (weighted = (positive-negative)/review_count):",
    ]
    # Every category goes in, low-mention ones included — the model gets a call-out per
    # category regardless (see _build_system_prompt), so it needs stats for all of them, not
    # just ones that clear the reporting threshold. low_mention flags the ones that don't, so
    # the model knows which categories need a label_caveat and a shorter write-up.
    for cat, stats in category_scores.items():
        low_mention_flag = " low_mention=true" if not stats["meets_threshold"] else ""
        lines.append(
            f"- {cat}: weighted={stats['weighted']:.2f}, positive={stats['positive']}, "
            f"mixed={stats['mixed']}, negative={stats['negative']}, "
            f"mention_rate={stats['mention_rate']:.0%}, "
            f"suggested_label={stats['suggested_label']}{low_mention_flag}"
        )

    lines.append("")
    lines.append("Quote pool (category | outlet | value | quote):")
    for cat, entries in data.items():
        if cat not in category_scores:
            continue
        for key, (value, quote) in entries.items():
            outlet = outlet_by_key.get(key, key)
            lines.append(f"- {cat} | {outlet} | {value:+d} | {quote}")

    if fan_reaction_notes and fan_reaction_notes.strip():
        lines.append("")
        lines.append(
            "Fan/user reaction notes (raw, producer-supplied — NOT part of the scored "
            "press-review data above; synthesize the Impression of Fan Reaction section from "
            "this only):"
        )
        lines.append(fan_reaction_notes.strip())

    return "\n".join(lines)


def draft_narrative(client, game_title: str, platforms: str, avg_score: float,
                     review_count: int, category_scores: dict, data: dict, reviews: list,
                     model: Optional[str] = None, max_tokens: int = 8000,
                     include_recommendations: bool = True,
                     fan_reaction_notes: Optional[str] = None) -> dict:
    """
    max_tokens defaults to 8000, not the original 4000 — confirmed against a real 35-review run
    with 10+ well-supported categories where the model filled in a rich Executive Summary and
    Press Reactions synthesis (the first two fields in the schema) and then returned completely
    EMPTY category_callouts and recommendations arrays, despite there plainly being enough
    reportable data for several call-outs. The likely mechanism: those two fields come first in
    the schema, the model fills them in generously, and by the time it reaches the later array
    fields on a large run it's run low on output budget — Claude still closes out the JSON
    validly rather than erroring, so this fails silently as "no significant findings" rather
    than as a visible error. Doubling the budget makes running out far less likely; the retry
    below is the backstop for whatever budget still isn't enough on an even larger run.

    include_recommendations=False leaves the "PD Recommendations for Next Title" section out of
    the report entirely (report.py omits the heading too when this list is empty) — the schema
    itself omits the field rather than asking the model to draft something just to discard it.

    fan_reaction_notes: raw producer-supplied fan/user commentary (forum notes, storefront user
    reviews, survey blurbs, etc.), or None/blank when this run has none. Whether the "Impression
    of Fan Reaction" section is drafted at all is derived from this — not a separate boolean —
    so there's no way to end up with a fan_reaction section drafted from nothing (Sega's own
    Puyo Puyo Tetris reference report explicitly calls out when a section like this doesn't
    apply rather than showing it empty; this follows the same principle).
    """
    from .classify import DEFAULT_MODEL
    model = model or DEFAULT_MODEL
    include_fan_reaction = bool(fan_reaction_notes and fan_reaction_notes.strip())
    all_categories = list(category_scores.keys())
    user_prompt = build_narrative_user_prompt(
        game_title, platforms, avg_score, review_count, category_scores, data, reviews,
        fan_reaction_notes=fan_reaction_notes,
    )
    tool = _build_narrative_tool(all_categories, include_recommendations, include_fan_reaction)
    system_prompt = _build_system_prompt(include_recommendations, include_fan_reaction)

    def _call(tokens: int) -> dict:
        msg = client.messages.create(
            model=model,
            max_tokens=tokens,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_narrative"},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == "submit_narrative":
                return block.input
        raise RuntimeError("Model did not return a submit_narrative tool call")

    result = _call(max_tokens)

    def _missing_categories(res: dict) -> set:
        returned = {co.get("category") for co in res.get("category_callouts", [])
                    if isinstance(co, dict)}
        return set(all_categories) - returned

    missing = _missing_categories(result)
    if all_categories and missing:
        # Every category is required to get a callout now (not just a "reportable" subset), so
        # ANY gap — not just a totally empty array — is the incomplete-output failure mode this
        # retry exists to catch (same root cause as before: the model runs low on output budget
        # partway through a long list of categories and stops). One retry at double the budget.
        result = _call(max_tokens * 2)
        missing = _missing_categories(result)

    if missing:
        # Still short after the retry — inject a minimal, honest fallback entry for whatever's
        # left rather than silently shipping a report that skips a category. This guarantees
        # every category the run actually scored gets SOME paragraph in the final report,
        # regardless of what the model did.
        callouts = result.setdefault("category_callouts", [])
        for cat in all_categories:
            if cat not in missing:
                continue
            stats = category_scores[cat]
            low_mention = not stats.get("meets_threshold")
            callouts.append({
                "category": cat,
                "label": stats.get("suggested_label", "Not enough data"),
                "label_caveat": (
                    "Based on a very small number of mentions — treat this label with caution."
                    if low_mention else ""
                ),
                "synthesis": (
                    f"Mentioned by {stats.get('mention_rate', 0):.0%} of reviews — not enough "
                    "discussion to synthesize further this run."
                ),
                "quotes": [],
            })

    if not include_recommendations:
        result["recommendations"] = []  # enforce regardless of what the model actually returned
    if not include_fan_reaction:
        result["fan_reaction"] = []  # same principle — no notes given in, nothing drafted out

    return result
