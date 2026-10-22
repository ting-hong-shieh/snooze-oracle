"""README charts: matplotlib + tueplots axes styling, Okabe-Ito colours, Inter on a navy card.
Wide layouts for the GitHub README (about 830 px column).

    python docs/figures/readme_charts.py   # needs tueplots and Inter-{Regular,SemiBold,Bold}.ttf here

Writes docs/img/results_wide.png and docs/img/curve_wide.png (and PDFs next to this script).
Numbers: docs/commute_backtest.md; the curve is read from the demo page's embedded data.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyBboxPatch
from tueplots import axes as tue_axes

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "docs" / "img"
for w in ("Regular", "SemiBold", "Bold"):
    fontManager.addfont(str(HERE / f"Inter-{w}.ttf"))
F = {w: FontProperties(fname=str(HERE / f"Inter-{w}.ttf")) for w in ("Regular", "SemiBold", "Bold")}

# Card and text tokens from the demo page.
BG1 = "#0d1430"
INK, INK2, INK3, LINE = "#eef1fb", "#b6bdd6", "#8189a8", "#262d4d"
# Okabe-Ito: sky blue for the model, yellow for the target, vermillion for the cutoff.
# Context series in one de-emphasis gray. Checked with the dataviz validator on BG1:
# sky blue vs gray normal-vision dE 20.9, CVD dE 19.2; every colour >= 3:1 on BG1.
SKY, YELLOW, VERMILLION, GRAY = "#56B4E9", "#F0E442", "#D55E00", "#6e7180"
W, R = 1600, 24
PX = 0.72  # source px -> pt at dpi 100

plt.rcParams.update({
    **tue_axes.lines(base_width=1.6),
    **tue_axes.tick_direction(x="out", y="out"),
    **tue_axes.spines(left=False, right=False, top=False, bottom=False),
    **tue_axes.grid(grid_alpha=1.0, grid_linestyle="solid"), "grid.linewidth": 1.2,
    "grid.color": LINE, "axes.facecolor": "none", "axes.edgecolor": LINE,
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelcolor": INK3, "ytick.labelcolor": INK3,
    "font.family": "InterPost", "font.size": 40 * PX,
})


def card(h):
    fig = plt.figure(figsize=(W / 100, h / 100), dpi=100)
    fig.patch.set_alpha(0)
    fig.patches.append(FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={R / W}",
                                      transform=fig.transFigure, fc=BG1, ec="none", zorder=-10,
                                      mutation_aspect=W / h))
    return fig


def ftext(fig, x, y, s, size, color=INK, w="Regular", **kw):
    """Text in figure pixels from the top-left."""
    W_, H_ = fig.get_size_inches() * fig.dpi
    return fig.text(x / W_, 1 - y / H_, s, color=color, fontsize=size * PX, fontproperties=F[w], va="center", **kw)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=100, transparent=True)
    fig.savefig(HERE / f"{name}.pdf", transparent=True)


def bars(ax, vals, rows, top_labels):
    rx, ry = 8 * 25 / 520, 8 * 3.2 / 230
    ys = [2, 1, 0]
    for y, v, (_, c, _, _) in zip(ys, vals, rows):
        ax.add_patch(FancyBboxPatch((0, y - 0.32), v, 0.64, boxstyle=f"round,pad=0,rounding_size={rx}",
                                    mutation_aspect=ry / rx, fc=c, ec="none", zorder=2))
        ax.add_patch(plt.Rectangle((0, y - 0.32), 2 * rx, 0.64, fc=c, ec="none", zorder=2))
    ax.axvline(10, color=YELLOW, lw=3, zorder=3)
    ax.set_xlim(0, 25); ax.set_ylim(-0.6, 2.6)
    ax.set_yticks(ys, [r[0] for r in rows] if top_labels else [])
    for lab, (_, _, ink, w) in zip(ax.get_yticklabels(), rows):
        lab.set_color(ink); lab.set_fontproperties(F[w]); lab.set_fontsize(34 * PX)
    ax.tick_params(axis="y", length=0, pad=18)
    ax.set_xticks([0, 10, 20], ["0%", "10%", "20%"])
    ax.tick_params(axis="x", length=0, pad=14, labelsize=32 * PX)
    ax.grid(axis="x", zorder=0); ax.grid(axis="y", visible=False)
    for y, v, (_, c, ink, _) in zip(ys, vals, rows):
        # value after the bar end; a navy knockout keeps it readable where it crosses the 10% line
        ax.text(v + 0.7, y, f"{v:.1f}%", color=ink, fontsize=36 * PX, fontproperties=F["Bold"], va="center", zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG1, ec="none"))


def results_wide():
    # Form: emphasis. TabPFN is the series the story is about; both fixed buffers are context.
    h = 650
    fig = card(h)
    ftext(fig, 48, 60, "Late mornings, August replay", 46, w="Bold")
    ftext(fig, 48, 112, "Share of mornings late, lower is better. Yellow line: the alarm's 10% target.", 30, INK2)
    rows = [("10-min buffer", GRAY, INK2, "Regular"), ("13-min buffer*", GRAY, INK2, "Regular"), ("TabPFN alarm", SKY, INK, "SemiBold")]
    for i, (title, vals) in enumerate((("All 12 routes", (8.3, 5.1, 5.1)), ("The demo route", (22.9, 17.1, 12.7)))):
        x0 = 290 + i * 640
        ftext(fig, x0, 190, title, 36, w="SemiBold")
        ax = fig.add_axes([x0 / W, 1 - 500 / h, 500 / W, 270 / h])
        bars(ax, vals, rows, top_labels=(i == 0))
    ftext(fig, 48, 608, "* chosen after seeing August", 28, INK3)
    save(fig, "results_wide")


def curve_wide():
    page = (REPO / "docs/demo/index.html").read_text()
    data = json.loads(re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>', page, re.S).group(1))
    sc = next(s for s in data["scenarios"] if s["id"] == "commute-235->161")
    c, plan = sc["curve"], sc["plan"]
    hh, mm = map(int, c["start"].split(":"))
    xs = [hh * 60 + mm + i for i in range(len(c["p"]))]
    ps = [p * 100 for p in c["p"]]
    p_at = dict(zip(xs, ps))
    hm = lambda s: int(s[:2]) * 60 + int(s[3:])
    m2s = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    wake, ring = hm(plan["wake_time"]), hm(plan["first_ring"])
    gw = hm(plan["alternatives"]["gut_rule"]["wake_time"])
    h = 640
    fig = card(h)
    ftext(fig, 48, 60, "Chance of being on time, by wake-up time", 46, w="Bold")
    ftext(fig, 48, 112, "Demo morning: 9:00 meeting, University Heights (Bronx) to Midtown Center. TabPFN, the night before.", 30, INK2)
    ax = fig.add_axes([150 / W, 1 - 560 / h, 1390 / W, 340 / h])
    ax.axvspan(ring, wake, color=YELLOW, alpha=0.08, lw=0, zorder=0)
    ax.axhline(90, color=YELLOW, lw=3, zorder=2)
    ax.plot(xs, ps, color=SKY, lw=4.5, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    kw = dict(mec=BG1, mew=4, zorder=6)
    ax.plot([ring, ring + 10, ring + 20], [p_at[m] for m in (ring, ring + 10, ring + 20)], "o", ms=13, color=INK, **kw)
    ax.plot([wake], [p_at[wake]], "o", ms=18, color=VERMILLION, **kw)
    ax.plot([gw], [p_at[gw]], "o", ms=15, color=GRAY, **kw)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.set_xticks(list(range(xs[0] - xs[0] % 15 + 15, xs[-1] + 1, 15)))
    ax.set_xticklabels([m2s(int(m)) for m in ax.get_xticks()])
    ax.tick_params(length=0, pad=14, labelsize=32 * PX)
    ax.tick_params(axis="x", pad=24)
    ax.grid(axis="y", zorder=0); ax.grid(axis="x", visible=False)
    ann = dict(fontsize=32 * PX, va="center", zorder=7)
    ax.annotate("rings 07:14", (ring, 100), xytext=(0, 34), textcoords="offset points", ha="center", color=INK2, fontproperties=F["Regular"], **ann)
    ax.annotate("snoozes", (ring + 15, 100), xytext=(0, 34), textcoords="offset points", ha="center", color=INK2, fontproperties=F["Regular"], **ann)
    ax.annotate(f"{m2s(wake)}: last safe, {round(p_at[wake])}%", (wake, p_at[wake]), xytext=(18, 46), textcoords="offset points",
                ha="left", color=INK, fontproperties=F["SemiBold"], arrowprops=dict(arrowstyle="-", color=VERMILLION, lw=2, shrinkA=0, shrinkB=9), **ann)
    ax.annotate(f"+10 min rule\n{m2s(gw)}, {round(p_at[gw])}%", (gw, p_at[gw]), xytext=(-26, -52), textcoords="offset points",
                ha="right", color=INK2, linespacing=1.3, arrowprops=dict(arrowstyle="-", color=GRAY, lw=2, shrinkA=0, shrinkB=8), fontproperties=F["SemiBold"], **ann)
    ax.text(xs[0] + 1, 82, "90% target", color=INK2, fontproperties=F["SemiBold"], ha="left", **ann)
    ax.text((ring + wake) / 2, 8, "snooze time", color=INK2, fontproperties=F["SemiBold"], ha="center", **ann)
    save(fig, "curve_wide")


if __name__ == "__main__":
    results_wide()
    curve_wide()
