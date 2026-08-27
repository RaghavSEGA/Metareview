"""
Renders the Opinion Graph (weighted per-category sentiment) as a static PNG — shared by the
xlsx matrix workbook (matrix.py) and the docx report (report.py), so both show the exact same,
pixel-identical picture regardless of what tool ends up opening them.

Why a static image instead of a native Excel chart object (which is what matrix.py used to
embed via openpyxl's BarChart): that native chart is well known to render incorrectly outside
Excel itself. Confirmed against a real generated workbook: opened in a non-Excel viewer, the
bars themselves showed with correct proportional lengths, but every category-axis label came
back completely blank — just an empty, oddly-rotated placeholder box where each label should
have been. This matches a long-standing openpyxl/Excel-chart interoperability gap (Google
Sheets, LibreOffice, and various online xlsx previewers are all known to mishandle openpyxl-
authored chart XML, particularly the category axis) — Excel desktop itself renders the exact
same chart correctly, but a producer's report gets opened in whatever's on hand, not
necessarily Excel desktop. A rendered picture has no such dependency: what matplotlib draws is
exactly what every viewer shows, with nothing left for a second renderer to get wrong.
"""
from io import BytesIO
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display server in this app's runtime (local or hosted)
import matplotlib.pyplot as plt

# Matches the on-screen Plotly preview's color_continuous_scale in streamlit_app.py (same
# --red/--bg1/--grn tokens as the app's CSS palette) — a producer sees this same red-navy-green
# gradient on-screen before downloading, so the xlsx/docx picture should look like the same
# chart, not a differently-styled one.
_NEGATIVE, _NEUTRAL, _POSITIVE = "#CC2244", "#1A2A4A", "#00BB66"


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lerp(a, b, t: float):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _color_for_weighted(value: float) -> str:
    """3-stop red -> navy -> green gradient across [-1, 1] — value <= 0 interpolates
    red-to-navy, value >= 0 interpolates navy-to-green, matching a diverging scale centered
    on 0 rather than a plain two-color split."""
    value = max(-1.0, min(1.0, value))
    neg, neu, pos = _hex_to_rgb(_NEGATIVE), _hex_to_rgb(_NEUTRAL), _hex_to_rgb(_POSITIVE)
    r, g, b = _lerp(neg, neu, value + 1) if value <= 0 else _lerp(neu, pos, value)
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def render_opinion_graph_png(game_title: str, scores: Dict[str, dict]) -> Optional[bytes]:
    """
    scores: {category: {"weighted": float, "meets_threshold": bool, ...}} — the same shape
    matrix.compute_weighted_scores() returns. Plots EVERY category given, not just ones that
    clear the reporting threshold — a category discussed by very few reviews still has real
    (if thin) data behind it, and hiding it entirely was more confusing than useful: a producer
    scanning the graph had no way to tell "not mentioned" apart from "excluded for being
    under-mentioned." Instead, a low-mention category (meets_threshold False) gets a "*" appended
    to its label and a footnote explaining what that marks — visible, not vanished.

    Sorted so the most positively-received category renders at the top and the most negative
    at the bottom (ascending weighted order, since matplotlib's barh draws index 0 at the
    bottom) — matches the on-screen Plotly preview's sort order in streamlit_app.py.

    Returns PNG bytes, or None if `scores` is empty — callers should skip the graph/section
    entirely in that case rather than embed a blank chart.
    """
    if not scores:
        return None

    ordered = sorted(scores.items(), key=lambda kv: kv[1]["weighted"])
    labels = [(f"{k} *" if not v.get("meets_threshold") else k) for k, v in ordered]
    values = [v["weighted"] for _, v in ordered]
    colors = [_color_for_weighted(v["weighted"]) for _, v in ordered]
    any_low_mention = any(not v.get("meets_threshold") for _, v in ordered)

    fig_height = max(2.5, 0.42 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_height), dpi=150)
    ax.barh(labels, values, color=colors)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="#999999", linewidth=0.8, zorder=0)
    ax.set_xlabel("Weighted score")
    ax.set_title(f"{game_title} — Opinion Graph (weighted)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if any_low_mention:
        fig.text(0.01, 0.01, "* mentioned by very few reviews — read with caution",
                  fontsize=8, color="#777777")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
