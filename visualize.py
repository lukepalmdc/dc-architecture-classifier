"""
visualize.py  —  Swiss/International style KDE plots
Architecture styles over year built.

Usage:
    python visualize.py
    python visualize.py --csv dc_results/buildings_classified.csv --out plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Swiss/International style constants ──────────────────────────────────────
FONT        = "Arial"          # Helvetica substitute; change to "Helvetica Neue" if available
BG          = "#FFFFFF"
TEXT        = "#111111"
GRID        = "#EBEBEB"
SPINE       = "#222222"
TICK        = "#444444"

# 16-colour palette — primary / functional, no pastels
PALETTE = [
    "#D62728", "#1F77B4", "#2CA02C", "#FF7F0E",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF", "#AEC7E8", "#FFBB78",
    "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94",
]

LABEL_DISPLAY = {
    "art_deco":           "Art Deco",
    "art_nouveau":        "Art Nouveau",
    "beaux_arts":         "Beaux-Arts",
    "brutalist":          "Brutalist",
    "colonial_revival":   "Colonial Revival",
    "craftsman":          "Craftsman",
    "federal":            "Federal",
    "gothic_revival":     "Gothic Revival",
    "greek_revival":      "Greek Revival",
    "midcentury_modern":  "Mid-Century Modern",
    "modernist":          "Modernist",
    "neoclassical":       "Neoclassical",
    "postmodern":         "Postmodern",
    "romanesque_revival": "Romanesque Revival",
    "tudor_revival":      "Tudor Revival",
    "victorian":          "Victorian",
}


def apply_swiss(ax, title="", subtitle=""):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)

    # Spines: only bottom
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.8)

    # Grid: horizontal only, very light
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Ticks
    ax.tick_params(axis="both", colors=TICK, labelsize=8, length=3, width=0.6)
    ax.xaxis.set_tick_params(labelbottom=True)

    # Title block — left aligned
    if title:
        ax.text(0, 1.08, title.upper(), transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=TEXT,
                fontfamily=FONT, ha="left", va="bottom")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=8, color="#555555", fontfamily=FONT,
                ha="left", va="bottom")


def load_data(csv_path, residential_type=None):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "classified"].copy()
    if residential_type:
        types = [" ".join(t.split()).upper() for t in residential_type.split(",")]
        df = df[df["residential_type"].fillna("").str.strip().str.upper().isin(types)]
    df = df.dropna(subset=["year_built", "predicted_label"])
    df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
    df = df.dropna(subset=["year_built"])
    df = df[(df["year_built"] >= 1800) & (df["year_built"] <= 2025)]
    df["label"] = df["predicted_label"].map(LABEL_DISPLAY).fillna(df["predicted_label"])
    return df


# =============================================================================
# PLOT 1 — Ridgeline KDE: one row per style
# =============================================================================

def plot_ridgeline(df, out_dir, title_prefix="Washington DC", suffix=""):
    styles  = sorted(df["label"].unique())
    n       = len(styles)
    palette = dict(zip(styles, PALETTE[:n]))

    overlap = 2.2
    fig, axes = plt.subplots(n, 1, figsize=(11, n * 0.9),
                             sharex=True, facecolor=BG)
    fig.subplots_adjust(hspace=-0.45)

    x_min, x_max = df["year_built"].min() - 5, df["year_built"].max() + 5

    for i, (style, ax) in enumerate(zip(styles, axes)):
        data   = df[df["label"] == style]["year_built"]
        color  = palette[style]
        count  = len(data)

        ax.set_facecolor(BG)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, None)

        if len(data) >= 5:
            sns.kdeplot(data, ax=ax, fill=True, color=color,
                        alpha=0.25, linewidth=0, bw_adjust=1.2)
            sns.kdeplot(data, ax=ax, fill=False, color=color,
                        alpha=0.9, linewidth=1.5, bw_adjust=1.2)

            # Median tick
            med = data.median()
            ylim = ax.get_ylim()
            ax.axvline(med, color=color, linewidth=0.8, alpha=0.6, linestyle="--")

        # Remove all spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel("")

        # Style label — right of plot
        ax.text(1.01, 0.3, style.upper(), transform=ax.transAxes,
                fontsize=7.5, fontweight="bold", color=color,
                fontfamily=FONT, ha="left", va="center")
        ax.text(1.01, -0.3, f"n={count:,}", transform=ax.transAxes,
                fontsize=6.5, color="#888888", fontfamily=FONT,
                ha="left", va="center")

        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

    # Bottom axis only on last
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].spines["bottom"].set_color(SPINE)
    axes[-1].spines["bottom"].set_linewidth(0.8)
    axes[-1].tick_params(axis="x", colors=TICK, labelsize=8, length=3)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(25))

    # Title block
    fig.text(0.0, 1.01, title_prefix.upper() + " — ARCHITECTURAL STYLES BY YEAR BUILT",
             fontsize=12, fontweight="bold", color=TEXT,
             fontfamily=FONT, ha="left", transform=axes[0].transAxes)
    fig.text(0.0, 0.955, "Kernel density estimate  ·  classified buildings only  ·  dashed line = median",
             fontsize=7.5, color="#666666",
             fontfamily=FONT, ha="left", transform=axes[0].transAxes)

    path = out_dir / f"ridgeline_styles_year{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200, facecolor=BG)
    print(f"Saved {path}")
    plt.close(fig)


# =============================================================================
# PLOT 2 — Overlapping KDE: all styles on one axis
# =============================================================================

def plot_overlay(df, out_dir, title_prefix="Washington DC", suffix=""):
    styles  = sorted(df["label"].unique())
    palette = dict(zip(styles, PALETTE[:len(styles)]))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)

    for style in styles:
        data = df[df["label"] == style]["year_built"]
        if len(data) < 5:
            continue
        color = palette[style]
        sns.kdeplot(data, ax=ax, fill=False, color=color,
                    alpha=0.85, linewidth=1.4, bw_adjust=1.2,
                    label=style)

    apply_swiss(ax,
                title=f"{title_prefix} — Architectural Styles by Year Built",
                subtitle="Kernel density estimate · classified buildings only")

    ax.set_xlabel("YEAR BUILT", fontsize=8, fontfamily=FONT,
                  color=TICK, labelpad=8, fontweight="bold")
    ax.set_ylabel("DENSITY", fontsize=8, fontfamily=FONT,
                  color=TICK, labelpad=8, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))

    legend = ax.legend(
        fontsize=7, frameon=False, ncol=2,
        loc="upper right", labelcolor=TEXT,
        prop={"family": FONT, "size": 7},
    )

    path = out_dir / f"overlay_styles_year{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200, facecolor=BG)
    print(f"Saved {path}")
    plt.close(fig)


# =============================================================================
# PLOT 3 — Building count by style (horizontal bar, sorted)
# =============================================================================

def plot_counts(df, out_dir, title_prefix="Washington DC", suffix=""):
    counts = (df.groupby("label")
                .size()
                .sort_values(ascending=True)
                .reset_index(name="count"))

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    ax.barh(counts["label"], counts["count"], color=colors,
            height=0.6, edgecolor="none")

    # Count labels
    for _, row in counts.iterrows():
        ax.text(row["count"] + counts["count"].max() * 0.005,
                row["label"], f'{row["count"]:,}',
                va="center", fontsize=7.5, color=TEXT, fontfamily=FONT)

    apply_swiss(ax,
                title=f"{title_prefix} — Buildings Classified by Architectural Style",
                subtitle="Classified buildings only")

    ax.set_xlabel("NUMBER OF BUILDINGS", fontsize=8, fontfamily=FONT,
                  color=TICK, labelpad=8, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, counts["count"].max() * 1.12)

    path = out_dir / f"counts_by_style{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200, facecolor=BG)
    print(f"Saved {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dc_results/buildings_classified.csv")
    p.add_argument("--out", default="plots")
    p.add_argument("--type", dest="residential_type", default=None,
                   help="Filter by residential_type, e.g. 'NON RESIDENTIAL', 'RESIDENTIAL', 'MIXED USE'")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv, residential_type=args.residential_type)

    if args.residential_type:
        types_list = [" ".join(t.split()) for t in args.residential_type.split(",")]
        label = " + ".join(t.title() for t in types_list)
        title_prefix = f"Washington DC — {label}"
        suffix = "_" + "_".join(t.lower().replace(" ", "_") for t in types_list)
    else:
        title_prefix = "Washington DC"
        suffix = ""

    print(f"Loaded {len(df):,} classified buildings across {df['label'].nunique()} styles")
    if len(df) == 0:
        raw = pd.read_csv(args.csv)
        classified = raw[raw["status"] == "classified"]
        print(f"Total classified rows: {len(classified)}")
        print("residential_type values among classified:", classified["residential_type"].str.strip().unique().tolist())
        return
    print(f"Year range: {int(df['year_built'].min())} – {int(df['year_built'].max())}")

    plot_ridgeline(df, out_dir, title_prefix=title_prefix, suffix=suffix)
    plot_overlay(df, out_dir, title_prefix=title_prefix, suffix=suffix)
    plot_counts(df, out_dir, title_prefix=title_prefix, suffix=suffix)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
