"""Build the paper's figures (paper/figures/*.pdf) and tables (paper/tables/*.tex)
from results/ and data/. Every number in a table comes from these files.

    uv run python -I scripts/make_paper_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathrisk.liquidation import liquidation_distance  # noqa: E402
from pathrisk.metrics import brownian_quantile_multiple  # noqa: E402
from pathrisk.pipeline import StudyConfig, load_universe  # noqa: E402

RES, DATA = ROOT / "results", ROOT / "data"
FIG, TAB = ROOT / "paper" / "figures", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TIERS = ["majors", "large", "tail", "new"]
LABEL = {"majors": "Majors (BTC, ETH)", "large": "Large (top 30)", "tail": "Tail", "new": "New ($<$30 days)"}
SHORT = {"majors": "Majors", "large": "Large", "tail": "Tail", "new": "New"}
# Reference categorical palette, slots 1-4 in fixed order (adjacent pairs validated, light mode).
# Every series also carries its own marker and a legend entry, so identity never rests on colour.
COLOR = {"majors": "#2a78d6", "large": "#eb6834", "tail": "#1baf7a", "new": "#eda100"}
MARKER = {"majors": "o", "large": "s", "tail": "^", "new": "D"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#9a9993"
CRITICAL = "#d03b3b"
H_LABEL = {60: "1 h", 240: "4 h", 480: "8 h", 1440: "24 h"}
LEVERAGES = [2, 3, 5, 10, 20, 25, 50]
EXAMPLE_LEVERAGES = [10, 25, 50]

# Academic style: Computer Modern, matching the LaTeX body text.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 9.5,
    "axes.labelsize": 9.5, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.direction": "out", "ytick.direction": "out", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "text.color": INK, "pdf.fonttype": 42,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
})

mark = pd.read_csv(RES / "mark" / "summary.csv")
last = pd.read_csv(RES / "last" / "summary.csv")
cohort = pd.read_csv(RES / "mark-cohort" / "summary.csv")
q_mark = pd.read_csv(RES / "mark" / "quantiles.csv")
q_last = pd.read_csv(RES / "last" / "quantiles.csv")
q_cohort = pd.read_csv(RES / "mark-cohort" / "quantiles.csv")
coverage = pd.read_csv(RES / "mark" / "universe_coverage.csv", index_col=0, parse_dates=True)
BROWN_Q99 = brownian_quantile_multiple(0.99)
SNAP = pd.read_csv(ROOT / "paper" / "data" / "venue_leverage.csv")
BIN = SNAP[(SNAP.venue == "Binance") & SNAP.first_bracket_mmr.notna()]
MMR_BTC = float(BIN.loc[BIN.contract == "BTCUSDT", "first_bracket_mmr"].iloc[0])


def pct(x, nd=1):
    return "--" if pd.isna(x) else f"{100 * x:.{nd}f}\\%"  # LaTeX tables need \\%


def panel(ax, letter, text):
    ax.text(0.0, 1.03, f"({letter}) {text}", transform=ax.transAxes, fontsize=9.5, va="bottom")


def pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))


def tier_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.2, columnspacing=1.6)


# ---------------------------------------------------------------- Figure 1: one real window, three leverages
def fig_example():
    sym, start, h, m = "BTCUSDT", pd.Timestamp("2026-02-05 17:00", tz="UTC"), 1440, MMR_BTC
    b = pd.read_parquet(DATA / "parquet" / "markPriceKlines" / "1m" / f"{sym}.parquet")
    t = pd.to_datetime(b["open_time"], unit="ms", utc=True)
    w = b[(t >= start) & (t < start + pd.Timedelta(minutes=h))].copy()
    w["hours"] = (pd.to_datetime(w["open_time"], unit="ms", utc=True) - start).dt.total_seconds() / 3600
    o, c = w["open"].iloc[0], w["close"].iloc[-1]
    rel = lambda p: 100 * (p / o - 1)  # noqa: E731

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.fill_between(w["hours"], rel(w["low"]), rel(w["high"]), color=COLOR["majors"], alpha=0.2, lw=0)
    ax.plot(w["hours"], rel(w["close"]), color=COLOR["majors"], lw=1.0)
    ax.axhline(0, color=INK2, lw=0.8)
    touches = {}
    for L, ls in zip(EXAMPLE_LEVERAGES, [(0, (6, 2)), (0, (3, 2)), (0, (1, 1.5))]):
        d = liquidation_distance(L, m, "long")
        ax.axhline(-100 * d, color=CRITICAL, lw=1.0, ls=ls)
        hit = w.loc[w["low"] <= o * (1 - d), "hours"]
        touches[L] = (d, float(hit.iloc[0]) if len(hit) else None)
        ax.text(24.3, -100 * d, f"{L}x  ({-100 * d:.1f}%)", va="center", fontsize=8.5, color=CRITICAL)
        if len(hit):
            ax.plot([hit.iloc[0]], [-100 * d], marker="o", ms=4.5, color=CRITICAL, mec="white", mew=1)
    lo = w.loc[w["low"].idxmin()]
    ax.annotate(f"low {rel(lo['low']):.1f}%", (lo["hours"], rel(lo["low"])), xytext=(8, -2),
                textcoords="offset points", fontsize=8.5, va="center")
    ax.annotate(f"close {rel(c):+.1f}%", (24, rel(c)), xytext=(-6, 8), textcoords="offset points",
                fontsize=8.5, ha="right")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel(f"Hours after entry ({start:%-d %B %Y}, {start:%H:%M} UTC)")
    ax.set_ylabel("Mark price relative to entry")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.savefig(FIG / "fig1_example.pdf")
    plt.close(fig)
    return {"entry": o, "close": c, "low": lo["low"], "touches": touches}


# ---------------------------------------------------------------- Figure 2: path vs close breach by leverage
def fig_path_vs_close():
    """Small multiples: rows are horizons, columns are tiers. Solid: liquidated along the path (p_P);
    dashed: ends beyond the level at the close (p_C); the band between them is what the close misses."""
    hs = (60, 1440)
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 4.3), sharex=True, sharey="row")
    for r, h in enumerate(hs):
        for c, tier in enumerate(TIERS):
            ax = axes[r, c]
            s = mark[(mark.tier == tier) & (mark.horizon == h) & (mark.side == "long")].sort_values("leverage")
            col = COLOR[tier]
            ax.fill_between(s.leverage, s.p_close, s.p_path, color=col, alpha=0.22, lw=0)
            ax.plot(s.leverage, s.p_path, color=col, marker=MARKER[tier], mec="white", mew=0.8)
            ax.plot(s.leverage, s.p_close, color=col, lw=1.2, ls=(0, (3, 1.5)))
            ax.set_xscale("log")
            ax.set_xticks([2, 5, 10, 25, 50], ["2x", "5x", "10x", "25x", "50x"])
            ax.minorticks_off()
            ax.tick_params(axis="x", labelsize=8)
            ax.set_ylim(0, 0.4 if h == 60 else 1.0)
            pct_axis(ax)
            ax.text(0.03, 0.97, f"{SHORT[tier]}, {H_LABEL[h]}", transform=ax.transAxes, fontsize=8.5,
                    va="top", ha="left")
        axes[r, 0].set_ylabel(f"Probability, {H_LABEL[h]} horizon")
    for ax in axes[1]:
        ax.set_xlabel("Leverage", fontsize=8.5)
    g = MUTED
    handles = [matplotlib.lines.Line2D([], [], color=g, lw=1.6, marker="o", mec="white", ms=4),
               matplotlib.lines.Line2D([], [], color=g, lw=1.2, ls=(0, (3, 1.5))),
               matplotlib.patches.Patch(color=g, alpha=0.3, lw=0)]
    labels = ["Liquidated along the path, $p_P$", "Ends beyond the level, $p_C$", "Unseen at the close"]
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig2_path_vs_close.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3: q99 multiple vs horizon
def fig_quantile_multiple():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    hs = [60, 240, 480, 1440]
    for ax, side, letter in zip(axes, ("long", "short"), "ab"):
        for tier in TIERS:
            q = q_mark[(q_mark.tier == tier) & (q_mark.side == side) & (q_mark.alpha == 0.99)].set_index("horizon")
            ax.plot(range(4), q.loc[hs, "multiple"].to_numpy(), color=COLOR[tier], marker=MARKER[tier],
                    mec="white", mew=0.8, label=LABEL[tier])
        ax.axhline(BROWN_Q99, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.text(3.15, BROWN_Q99 + 0.01, "Brownian", fontsize=8, ha="right", va="bottom", color=INK2)
        ax.set_xticks(range(4), [H_LABEL[h] for h in hs])
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(1.0, 1.75)
        ax.set_xlabel("Holding horizon $h$")
        panel(ax, letter, f"{side} positions")
    axes[0].set_ylabel(r"$Q_{0.99}(\mathrm{path}) \,/\, Q_{0.99}(\mathrm{close})$")
    tier_legend(fig, axes[0])
    fig.tight_layout(w_pad=2.5, rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig3_quantile_multiple.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4: mark vs traded prices
def fig_mark_vs_last():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    x = np.arange(len(TIERS))
    wbar = 0.36
    for ax, h, letter in zip(axes, (60, 1440), "ab"):
        for k, (name, s, col) in enumerate((("Mark price", mark, "#2a78d6"), ("Last traded price", last, "#eb6834"))):
            v = s[(s.horizon == h) & (s.side == "long") & (s.leverage == 10)].set_index("tier").reindex(TIERS)
            vals = v.hidden_share.where(v.p_close > 0)
            bars = ax.bar(x + (k - 0.5) * (wbar + 0.02), vals.fillna(0), wbar, color=col, label=name)
            for rect, val in zip(bars, vals):
                if pd.notna(val):
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{100 * val:.0f}", ha="center",
                            va="bottom", fontsize=7.5, color=INK2)
                else:
                    ax.text(rect.get_x() + rect.get_width() / 2, 0.02, "n/a", ha="center", va="bottom",
                            fontsize=7, color=MUTED)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(x, [SHORT[t] for t in TIERS])
        ax.set_ylim(0, 0.85)
        pct_axis(ax)
        ax.grid(axis="x", visible=False)
        panel(ax, letter, f"10x long, {H_LABEL[h]} horizon")
    axes[0].set_ylabel("Unseen share $U$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=2.5, rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig4_mark_vs_last.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5: universe over time
def fig_universe():
    cov = coverage.reindex(columns=TIERS, fill_value=0)
    weekly = cov.resample("W").mean()
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.stackplot(weekly.index, *[weekly[t] for t in TIERS], colors=[COLOR[t] for t in TIERS],
                 labels=[LABEL[t] for t in TIERS], edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Contracts in the universe")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=4, fontsize=8.5)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "fig5_universe.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- tables
def table_data():
    symbols = len((DATA / "symbols.txt").read_text().split())
    listing = pd.read_csv(DATA / "listing.csv")
    n_mark = len(list((DATA / "parquet" / "markPriceKlines" / "1m").glob("*.parquet")))
    u = load_universe(DATA, StudyConfig(start="2025-11-01", end="2026-10-31"))
    end = u["trading_until"].max()
    delisted = u.loc[u["trading_until"] < end - pd.Timedelta(days=1), "symbol"].nunique()
    per_day = coverage.reindex(columns=TIERS, fill_value=0).mean()
    cohort_n = len((DATA / "cohort.txt").read_text().split())
    w = mark[(mark.leverage == 10) & (mark.side == "long")].groupby("horizon")["n_windows"].sum()
    rows = [
        ("USDT-margined perpetuals in the archive (delisted included)", f"{symbols:,}"),
        ("\\quad with a listing date", f"{listing['listing_date'].notna().sum():,}"),
        ("\\quad with 1-minute mark-price data in the study period", f"{n_mark:,}"),
        ("\\quad in the point-in-time universe on at least one day", f"{u['symbol'].nunique():,}"),
        ("\\quad of which delisted during the study", f"{delisted:,}"),
        ("Balanced cohort (live for the whole study)", f"{cohort_n:,}"),
        ("Mean contracts per day: majors / large / tail / new", " / ".join(f"{per_day[t]:.0f}" for t in TIERS)),
        ("Holding windows (1 h / 4 h / 8 h / 24 h), millions",
         " / ".join(f"{w[h] / 1e6:.2f}" for h in (60, 240, 480, 1440))),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    (TAB / "tab_data.tex").write_text(
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\nItem & Value \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def _unseen_cell(r):
    if r.p_close <= 0:
        return "--"
    return f"{pct(r.hidden_share, 0)} \\scriptsize[{pct(r.hidden_share_lo, 0)}, {pct(r.hidden_share_hi, 0)}]"


def table_breach():
    lines = []
    for tier in TIERS:
        for h in (60, 480, 1440):
            cells = [SHORT[tier] if h == 60 else "", H_LABEL[h]]
            for side in ("long", "short"):
                r = mark[(mark.tier == tier) & (mark.horizon == h) & (mark.side == side) & (mark.leverage == 10)].iloc[0]
                cells += [pct(r.p_close, 2), pct(r.p_path, 2), _unseen_cell(r)]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    (TAB / "tab_breach.tex").write_text(
        "\\begin{tabular}{@{}llrrlrrl@{}}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{Long, 10x} & \\multicolumn{3}{c}{Short, 10x} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "Tier & $h$ & $p_C$ & $p_P$ & $U$ [95\\% CI] & $p_C$ & $p_P$ & $U$ [95\\% CI] \\\\\n\\midrule\n"
        + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def offered(tier, L, h=60, side="long"):
    """Share of the tier's windows in which the contract offers leverage L (vs 2x, offered by all)."""
    n = mark[(mark.tier == tier) & (mark.horizon == h) & (mark.side == side)].set_index("leverage")["n_windows"]
    return n.get(float(L), 0) / n.get(2.0, np.nan)


def table_leverage():
    """10x, 25x and 50x side by side: coverage, close and path breach probabilities, unseen share."""
    lines = []
    for tier in TIERS:
        for i, L in enumerate(EXAMPLE_LEVERAGES):
            cells = [SHORT[tier] if i == 0 else "", f"{L}x", pct(offered(tier, L), 0)]
            for h in (60, 1440):
                r = mark[(mark.tier == tier) & (mark.horizon == h) & (mark.side == "long") & (mark.leverage == L)]
                if r.empty:
                    cells += ["--"] * 3
                    continue
                r = r.iloc[0]
                cells += [pct(r.p_close, 2), pct(r.p_path, 2), pct(r.hidden_share, 0) if r.p_close > 0 else "--"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    (TAB / "tab_leverage.tex").write_text(
        "\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
        " & & & \\multicolumn{3}{c}{1 h} & \\multicolumn{3}{c}{24 h} \\\\\n"
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
        "Tier & $L$ & Offered & $p_C$ & $p_P$ & $U$ & $p_C$ & $p_P$ & $U$ \\\\\n\\midrule\n"
        + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def table_quantiles():
    lines = []
    for tier in TIERS:
        for side in ("long", "short"):
            cells = [SHORT[tier] if side == "long" else "", side.capitalize()]
            for h in (60, 1440):
                r = q_mark[(q_mark.tier == tier) & (q_mark.side == side) & (q_mark.horizon == h) & (q_mark.alpha == 0.99)].iloc[0]
                cells += [pct(r.q_close, 2), pct(r.q_path, 2), f"{r.multiple:.2f}"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    lines.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Brownian benchmark}} & & & {BROWN_Q99:.3f} & & & {BROWN_Q99:.3f} \\\\")
    (TAB / "tab_quantiles.tex").write_text(
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{1 h} & \\multicolumn{3}{c}{24 h} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "Tier & Side & $Q_{0.99}^{C}$ & $Q_{0.99}^{P}$ & $\\kappa_{0.99}$ & $Q_{0.99}^{C}$ & $Q_{0.99}^{P}$ & $\\kappa_{0.99}$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_robustness():
    lines = []
    for tier in TIERS:
        cells = [SHORT[tier]]
        for q in (q_mark, q_cohort, q_last):
            r = q[(q.tier == tier) & (q.side == "long") & (q.horizon == 60) & (q.alpha == 0.99)]
            cells.append(f"{r.multiple.iloc[0]:.2f}" if len(r) else "--")
        for s in (mark, cohort, last):
            r = s[(s.tier == tier) & (s.side == "long") & (s.horizon == 60) & (s.leverage == 10)]
            cells.append(pct(r.hidden_share.iloc[0], 0) if len(r) and r.p_close.iloc[0] > 0 else "--")
        for q in (q_mark, q_cohort, q_last):
            r = q[(q.tier == tier) & (q.side == "short") & (q.horizon == 1440) & (q.alpha == 0.99)]
            cells.append(f"{r.multiple.iloc[0]:.2f}" if len(r) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    head = " & ".join(["Mark", "Cohort", "Last"] * 3)
    (TAB / "tab_robustness.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n"
        " & \\multicolumn{3}{c}{$\\kappa_{0.99}$, long, 1 h} & \\multicolumn{3}{c}{$U$, 10x long, 1 h}"
        " & \\multicolumn{3}{c}{$\\kappa_{0.99}$, short, 24 h} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        f"Tier & {head} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_liquidation():
    levs = [10, 20, 25, 50, 75, 100, 150]
    g = BIN.groupby("first_bracket_mmr").agg(n=("contract", "size"), lev=("max_leverage", "median"))
    g = g[g.n >= 2]  # keeps BTC and ETH (0.4%)
    lines = []
    for m, row in g.iterrows():
        cells = [pct(m, 2), f"{row.n:.0f}", f"{row.lev:.0f}x"]
        for L in levs:
            d = liquidation_distance(L, m, "long")
            cells.append(f"{100 * d:.2f}" if d > 0 else "--")
        lines.append(" & ".join(cells) + " \\\\")
    (TAB / "tab_liquidation.tex").write_text(
        "\\begin{tabular}{@{}rrr" + "r" * len(levs) + "@{}}\n\\toprule\n"
        "$m$ & Contracts & Max lev. & " + " & ".join(f"{L}x" for L in levs) + " \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def text_macros(ex) -> str:
    """Every number quoted in the prose, so the text can never drift from the tables."""
    nm = mark[mark.tier != "majors"]
    def u_range(df, L):
        x = df[(df.leverage == L) & (df.p_close > 0)].hidden_share
        return f"{100 * x.min():.0f}", f"{100 * x.max():.0f}"
    def cell(tier, h, side, L, col, s=mark):
        r = s[(s.tier == tier) & (s.horizon == h) & (s.side == side) & (s.leverage == L)]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    def p(x):
        return pct(x, 2 if x < 0.01 else 1)
    q99 = q_mark[q_mark.alpha == 0.99]
    BROWNIAN_K99 = float(q99.brownian_multiple.iloc[0])
    MMR_MED = float(BIN.first_bracket_mmr.median())
    def qcell(tier, h, side):
        return q99[(q99.tier == tier) & (q99.horizon == h) & (q99.side == side)].iloc[0]
    lev_cells = mark[(mark.tier != "majors") & (mark.side == "long") & mark.leverage.isin(EXAMPLE_LEVERAGES)
                     & mark.horizon.isin([60, 1440]) & (mark.p_close > 0)].hidden_share
    # cohort vs main: largest absolute change over every cell both runs define
    k = q99.merge(q_cohort[q_cohort.alpha == 0.99], on=["tier", "horizon", "side", "alpha"], suffixes=("", "_c"))
    uu = mark[(mark.leverage == 10) & (mark.p_close > 0)].merge(
        cohort[(cohort.leverage == 10) & (cohort.p_close > 0)], on=["tier", "horizon", "leverage", "side"], suffixes=("", "_c"))
    lo, hi = pd.Timestamp("2025-11-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC")
    lst = pd.read_csv(DATA / "listing.csv"); lst["listing_date"] = pd.to_datetime(lst["listing_date"], utc=True)
    lt = pd.read_csv(DATA / "last_trade.csv"); lt["trading_until"] = pd.to_datetime(lt["trading_until"], utc=True)
    new = lst[(lst.listing_date >= lo) & (lst.listing_date < hi)]
    gone = lt[(lt.trading_until >= lo) & (lt.trading_until < hi - pd.Timedelta(hours=1))]
    months = pd.period_range("2025-11", "2026-10", freq="M").strftime("%Y-%m")
    m_new = new.groupby(new.listing_date.dt.strftime("%Y-%m")).size().reindex(months, fill_value=0)
    m_gone = gone.groupby(gone.trading_until.dt.strftime("%Y-%m")).size().reindex(months, fill_value=0)
    used = pd.read_csv(RES / "mark" / "margins_used.csv")
    names = {10: "EXTEN", 25: "EXTWENTYFIVE", 50: "EXFIFTY"}
    m = {names[L]: f"{hit:.1f}" for L, (_, hit) in ex["touches"].items()}
    m.update({
        "UTenLo": u_range(nm, 10)[0], "UTenHi": u_range(nm, 10)[1],
        "UTwentyFiveLo": u_range(nm, 25)[0], "UTwentyFiveHi": u_range(nm, 25)[1],
        "TailDayLongPC": p(cell("tail", 1440, "long", 10, "p_close")),
        "TailDayLongPP": p(cell("tail", 1440, "long", 10, "p_path")),
        "TailDayLongPCTwentyFive": p(cell("tail", 1440, "long", 25, "p_close")),
        "TailDayLongPPTwentyFive": p(cell("tail", 1440, "long", 25, "p_path")),
        "TailDayShortPC": p(cell("tail", 1440, "short", 10, "p_close")),
        "TailDayShortPP": p(cell("tail", 1440, "short", 10, "p_path")),
        "MajorsDayU": f"{100 * cell('majors', 1440, 'long', 10, 'hidden_share'):.0f}",
        "TailHourPPTen": p(cell("tail", 60, "long", 10, "p_path")),
        "TailHourPPTwentyFive": p(cell("tail", 60, "long", 25, "p_path")),
        "TailHourPPFifty": p(cell("tail", 60, "long", 50, "p_path")),
        "TailHourPCTen": p(cell("tail", 60, "long", 10, "p_close")),
        "TailHourPCTwentyFive": p(cell("tail", 60, "long", 25, "p_close")),
        "TailHourPCFifty": p(cell("tail", 60, "long", 50, "p_close")),
        "TailDayPPFifty": p(cell("tail", 1440, "long", 50, "p_path")),
        "TailDayPCFifty": p(cell("tail", 1440, "long", 50, "p_close")),
        "LargeDayPPTwentyFive": p(cell("large", 1440, "long", 25, "p_path")),
        "LargeDayPCTwentyFive": p(cell("large", 1440, "long", 25, "p_close")),
        "LevULo": f"{100 * lev_cells.min():.0f}", "LevUHi": f"{100 * lev_cells.max():.0f}",
        "KappaLo": f"{q99.multiple.min():.2f}", "KappaHi": f"{q99.multiple.max():.2f}",
        "LargeHourQC": pct(q99[(q99.tier == "large") & (q99.horizon == 60) & (q99.side == "long")].q_close.iloc[0], 1),
        "LargeHourQP": pct(q99[(q99.tier == "large") & (q99.horizon == 60) & (q99.side == "long")].q_path.iloc[0], 1),
        "RWExLo": f"{100 * (q99.multiple.min() / BROWNIAN_K99 - 1):.0f}",
        "RWExHi": f"{100 * (q99.multiple.max() / BROWNIAN_K99 - 1):.0f}",
        "LongKappaExLo": f"{100 * (q99[q99.side == 'long'].multiple.min() - 1):.0f}",
        "LongKappaExHi": f"{100 * (q99[q99.side == 'long'].multiple.max() - 1):.0f}",
        "ShortKappaExHi": f"{100 * (q99[q99.side == 'short'].multiple.max() - 1):.0f}",
        "NewDayShortQC": pct(qcell("new", 1440, "short").q_close, 1),
        "NewDayShortQP": pct(qcell("new", 1440, "short").q_path, 1),
        "LevCloseLargeHour": f"{1 / (qcell('large', 60, 'long').q_close + MMR_MED):.0f}",
        "LevPathLargeHour": f"{1 / (qcell('large', 60, 'long').q_path + MMR_MED):.0f}",
        "LevCloseTailDayShort": f"{1 / (qcell('tail', 1440, 'short').q_close + MMR_MED):.1f}",
        "LevPathTailDayShort": f"{1 / (qcell('tail', 1440, 'short').q_path + MMR_MED):.1f}",
        "CohortMaxDK": f"{(k.multiple - k.multiple_c).abs().max():.2f}",
        "CohortMaxDU": f"{100 * (uu.hidden_share - uu.hidden_share_c).abs().max():.0f}",
        "RobLargeMark": f"{100 * cell('large', 60, 'long', 10, 'hidden_share'):.0f}",
        "RobLargeLast": f"{100 * cell('large', 60, 'long', 10, 'hidden_share', last):.0f}",
        "RobTailMark": f"{100 * cell('tail', 60, 'long', 10, 'hidden_share'):.0f}",
        "RobTailLast": f"{100 * cell('tail', 60, 'long', 10, 'hidden_share', last):.0f}",
        "MMRBTC": pct(MMR_BTC, 1), "MMRMedian": pct(BIN.first_bracket_mmr.median(), 1),
        "MMRMax": pct(BIN.first_bracket_mmr.max(), 0),
        "FallbackN": f"{used.loc[used.source == 'tier median', 'symbol'].nunique()}",
        "ListingsN": f"{len(new)}", "DelistingsN": f"{len(gone)}",
        "ListingsPerMonth": f"{m_new.mean():.1f}", "DelistingsPerMonth": f"{m_gone.mean():.1f}",
        "ListingsMin": f"{m_new.min()}", "ListingsMax": f"{m_new.max()}",
        "DelistingsMin": f"{m_gone.min()}", "DelistingsMax": f"{m_gone.max()}",
    })
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


if __name__ == "__main__":
    ex = fig_example()
    fig_path_vs_close()
    fig_quantile_multiple()
    fig_mark_vs_last()
    fig_universe()
    table_data()
    table_breach()
    table_leverage()
    table_quantiles()
    table_robustness()
    table_liquidation()
    (TAB / "macros.tex").write_text(text_macros(ex))
    print("example:", ex)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
