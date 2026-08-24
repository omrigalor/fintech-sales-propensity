"""
Render a professional flowchart of the model pipeline -> architecture.png (used in README + docs).
Run:  python make_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

NAVY, ACCENT, INK, MUTED = "#141b34", "#e8552c", "#1a2550", "#6b7280"
BLUE, GREEN, GREY, GOLD = "#eef2ff", "#e7f6ee", "#eceef3", "#fff5e6"

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 12); ax.set_ylim(0, 14.6); ax.axis("off")


def box(cx, cy, w, h, lines, fc=BLUE, ec="#c7cede", tc=INK, fs=9.5, bold_first=True, round=0.12):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle=f"round,pad=0.02,rounding_size={round}",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3))
    if isinstance(lines, str): lines = [lines]
    n = len(lines); y0 = cy + (n-1)*0.16
    for i, ln in enumerate(lines):
        ax.text(cx, y0 - i*0.32, ln, ha="center", va="center", fontsize=fs if i else fs+0.5,
                fontweight="bold" if (i == 0 and bold_first) else "normal",
                color=tc, zorder=4)


def band(y0, y1, label, color="#f4f5f8"):
    ax.add_patch(plt.Rectangle((0.2, y0), 11.6, y1-y0, facecolor=color, edgecolor="none", zorder=1))
    ax.text(0.45, y1-0.28, label, ha="left", va="center", fontsize=9, fontweight="bold", color=MUTED, zorder=2)


def arrow(x0, y0, x1, y1, color=NAVY, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=color, zorder=2, shrinkA=2, shrinkB=2))


# title
ax.text(6, 14.25, "Meridian — Model Pipeline", ha="center", fontsize=15, fontweight="bold", color=NAVY)

# INPUTS
band(13.0, 13.9, "INPUTS")
box(3.4, 13.45, 4.4, 0.7, ["2,000 companies", "(messy website_text)"], fc="#fff", fs=9)
box(8.6, 13.45, 4.4, 0.7, ["100 labeled won / lost", "(50 / 50)"], fc="#fff", fs=9)

# ENRICH
band(10.7, 12.7, "ENRICH — 5 evidence sources   (reasoning-pro on every LLM stage; scrape = regex)")
ex = [2.4, 4.35, 6.3, 8.25, 10.2]
en = [("enrich.py", "site → AR/AP-need"), ("scrape.py", "technographics"),
      ("external.py", "PE / multi-entity"), ("verify.py", "critic (35% agree)"),
      ("consistency.py", "self-consistency")]
for x, (a, b) in zip(ex, en):
    box(x, 11.55, 1.85, 0.95, [a, b], fc=GREEN, ec="#a9dcc0", fs=8.2)
arrow(6, 13.05, 6, 12.75)                       # inputs -> enrich band
for x in ex: arrow(x, 11.05, 6, 10.15, color="#b7bdcc", lw=1.1)  # sources -> reconcile

# RECONCILE
box(6, 9.75, 7.6, 0.7, ["RECONCILE  —  authority-by-field × confidence   (disagreement → flag, don't average)"],
    fc=GOLD, ec="#e8c98a", fs=9)
arrow(6, 9.4, 6, 8.95)

# SCORE
band(6.15, 8.75, "SCORE  (transparent ML — the LLM extracts, the ML decides)")
box(2.5, 7.75, 2.2, 0.75, ["FIT", "need · B2B · not-too-big"], fc=BLUE, fs=8.6)
box(6.0, 7.75, 2.6, 0.75, ["WIN  (calibrated GB)", "+ conformal abstention"], fc=BLUE, ec=ACCENT, fs=8.6)
box(9.5, 7.75, 2.2, 0.75, ["VALUE", "mid-mkt + PE boost"], fc=BLUE, fs=8.6)
box(6.0, 6.45, 5.6, 0.7, ["PRIORITY = blend( FIT × WIN × VALUE )  ·  rank-stability  ·  segments"],
    fc=NAVY, ec=NAVY, tc="#fff", fs=9)
for x in (2.5, 6.0, 9.5): arrow(x, 7.35, 6.0, 6.85, color="#b7bdcc", lw=1.1)
# trigger (separate timing lens)
box(10.55, 6.45, 2.5, 0.85, ["TRIGGER  (why-now)", "hiring · funding ·", "on-a-competitor"], fc=GOLD, ec="#e8552c", fs=8)
ax.text(10.55, 5.85, "separate timing lens", ha="center", fontsize=7, style="italic", color=MUTED)

# ranked_output
arrow(6, 6.1, 6, 5.55)
box(6, 5.15, 5.0, 0.68, ["ranked_output.csv   (all 2,000 scored & ranked)"], fc="#fff", ec=NAVY, fs=9.5)
arrow(6, 4.8, 6, 4.35)

# APP
band(2.3, 4.15, "app.py  —  one screen: who to call first  (local, instant; reads pre-computed outputs)")
box(4.1, 3.15, 2.8, 1.0, ["Ranked call list", "search · filter", "strategy dial · 🔥 signals"], fc="#fff", ec="#c7cede", fs=8)
box(7.9, 3.15, 2.8, 1.0, ["Account card", "why this account (LLM)", "signals + suggested opener"], fc="#fff", ec="#c7cede", fs=8)
arrow(6, 4.35, 4.1, 3.68, color="#b7bdcc", lw=1.0)
arrow(6, 4.35, 7.9, 3.68, color="#b7bdcc", lw=1.0)

# TECHNIQUES grounding footer
band(1.25, 2.35, "TECHNIQUES GROUNDING   (detail in README · WRITEUP)")
ax.text(6, 1.92, "Pydantic-validated extraction   ·   LLM-as-judge critic pass   ·   self-consistency (Wang 2022)",
        ha="center", fontsize=8.7, color=INK, zorder=4)
ax.text(6, 1.55, "conformal abstention (Angelopoulos & Bates 2021)   ·   Platt-calibrated GB + repeated 5×10 CV   ·   bootstrap rank-stability",
        ha="center", fontsize=8.7, color=INK, zorder=4)

fig.tight_layout()
out = Path(__file__).resolve().parent / "architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
