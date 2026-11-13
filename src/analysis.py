"""
analysis.py - reusable tables and charts for the notebook.

Every function takes the processed DataFrames from prepare.py and returns
either a tidy table or a matplotlib Figure, so the notebook stays readable.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
FIGURES = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#a3a29c"
# fixed colour per category, so a category is the same colour in every chart
CATEGORY_COLORS = {
    "Anime": BLUE,
    "Western animation": ORANGE,
    "Live-action drama": AQUA,
    "Live-action comedy": YELLOW,
}
RATING_CMAP = LinearSegmentedColormap.from_list(
    "rating", ["#fbe3e2", "#f3f2ee", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
    })


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=160, bbox_inches="tight")


def load():
    series = pd.read_parquet(os.path.join(PROCESSED, "series.parquet"))
    episodes = pd.read_parquet(os.path.join(PROCESSED, "episodes.parquet"))
    return series, episodes


def find(series, title, year=None):
    """tconst of the most-voted series with this exact title (optionally starting in `year`)."""
    m = series[series["primaryTitle"].str.lower() == title.lower()]
    if year:
        m = m[m["startYear"] == year]
    if m.empty:
        raise KeyError(title)
    return m.sort_values("numVotes", ascending=False).iloc[0]["tconst"]


# ---------------------------------------------------------------------------
# 1. categories
# ---------------------------------------------------------------------------
def category_summary(series, min_votes=1000):
    pop = series[series["numVotes"] >= min_votes]
    t = pop.groupby("category")["averageRating"].describe(percentiles=[.25, .5, .75])
    return t[["count", "25%", "50%", "75%", "mean"]].rename(columns={"50%": "median"}) \
        .sort_values("median").round(2)


def plot_category_ratings(series, min_votes=1000):
    t = category_summary(series, min_votes)
    t = t.drop(index="Other", errors="ignore")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(t))
    ax.hlines(y, t["25%"], t["75%"], color=BLUE, linewidth=6, alpha=0.35)
    ax.scatter(t["median"], y, color=BLUE, s=60, zorder=3)
    for yi, (cat, r) in zip(y, t.iterrows()):
        ax.text(r["75%"] + 0.05, yi, f"{r['median']:.1f}  (n={int(r['count']):,})",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(y, t.index)
    ax.set_xlim(t["25%"].min() - 0.3, t["75%"].max() + 0.9)
    ax.set_xlabel("Series rating (dot = median, bar = middle 50% of shows)")
    ax.set_title(f"How each kind of show rates (series with {min_votes:,}+ votes)", loc="left")
    ax.grid(axis="y", visible=False)
    return fig


def by_decade(series, min_votes=1000):
    pop = series[(series["numVotes"] >= min_votes) & series["category"].isin(CATEGORY_COLORS)]
    pop = pop.assign(decade=(pop["startYear"] // 10 * 10))
    t = pop.groupby(["decade", "category"]).agg(median=("averageRating", "median"), n=("tconst", "count")).reset_index()
    return t[(t["n"] >= 15) & (t["decade"] >= 1960)]


def plot_by_decade(series, min_votes=1000):
    t = by_decade(series, min_votes)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cat, color in CATEGORY_COLORS.items():
        d = t[t["category"] == cat]
        ax.plot(d["decade"], d["median"], color=color, linewidth=2, marker="o", markersize=5, label=cat)
    ax.set_xticks(sorted(t["decade"].unique()), [f"{int(d)}s" for d in sorted(t["decade"].unique())])
    ax.set_xlim(t["decade"].min() - 3, t["decade"].max() + 3)
    ax.set_ylabel("Median series rating")
    ax.set_title("Median rating by the decade a show premiered", loc="left")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_ylim(t["median"].min() - 0.05, t["median"].max() + 0.15)
    return fig


# ---------------------------------------------------------------------------
# 2. season-over-season
# ---------------------------------------------------------------------------
def season_table(series, episodes, min_votes=5000, min_eps=4, min_seasons=3):
    """Average episode rating per season, for shows with enough seasons/episodes to compare."""
    keep = series.loc[series["numVotes"] >= min_votes, "tconst"]
    e = episodes[episodes["parentTconst"].isin(keep)]
    ss = e.groupby(["parentTconst", "seasonNumber"]).agg(
        rating=("averageRating", "mean"), episodes=("tconst", "count")).reset_index()
    ss = ss[ss["episodes"] >= min_eps]
    shape = ss.groupby("parentTconst")["seasonNumber"].agg(["min", "count"])
    ok = shape[(shape["min"] == 1) & (shape["count"] >= min_seasons)].index
    ss = ss[ss["parentTconst"].isin(ok)].copy()
    s1 = ss[ss["seasonNumber"] == 1].set_index("parentTconst")["rating"]
    ss["vs_season_1"] = ss["rating"] - ss["parentTconst"].map(s1)
    return ss.merge(series[["tconst", "primaryTitle", "category"]], left_on="parentTconst", right_on="tconst") \
             .drop(columns="tconst")


def trajectory(ss, max_season=8, min_shows=15):
    t = ss[ss["category"].isin(CATEGORY_COLORS) & (ss["seasonNumber"] <= max_season)] \
        .groupby(["category", "seasonNumber"])["vs_season_1"].agg(["median", "count"]).reset_index()
    return t[t["count"] >= min_shows]


def plot_trajectory(ss, max_season=8):
    t = trajectory(ss, max_season)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(0, color=INK_2, linewidth=1)
    for cat, color in CATEGORY_COLORS.items():
        d = t[t["category"] == cat]
        ax.plot(d["seasonNumber"], d["median"], color=color, linewidth=2, marker="o", markersize=5, label=cat)
    ax.set_xlabel("Season")
    ax.set_ylabel("Median change vs season 1 (points)")
    ax.set_title("Do shows get better or worse as they go on?", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xticks(range(1, max_season + 1))
    return fig


# ---------------------------------------------------------------------------
# 3. pilots and finales
# ---------------------------------------------------------------------------
def pilot_finale(series, episodes, min_votes=10000, min_eps=10):
    ended = series[series["endYear"].notna() & (series["numVotes"] >= min_votes)
                   & (series["rated_episodes"] >= min_eps) & (series["titleType"] == "tvSeries")]
    e = episodes[episodes["parentTconst"].isin(ended["tconst"])] \
        .sort_values(["parentTconst", "seasonNumber", "episodeNumber"])
    g = e.groupby("parentTconst")
    t = pd.DataFrame({
        "pilot": g["averageRating"].first(),
        "finale": g["averageRating"].last(),
        "show_avg": g["averageRating"].mean().round(2),
    })
    t["finale_minus_pilot"] = (t["finale"] - t["pilot"]).round(1)
    t["finale_minus_avg"] = (t["finale"] - t["show_avg"]).round(2)
    return t.join(series.set_index("tconst")[["primaryTitle", "category", "startYear", "endYear", "numVotes"]])


LABEL_SHOWS = ["Game of Thrones", "Dexter", "How I Met Your Mother", "House of Cards", "Breaking Bad",
               "The Office", "Friends", "Parks and Recreation", "The Sopranos", "Seinfeld", "Death Note"]


def plot_pilot_finale(pf, label_min_votes=300000):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(pf["pilot"], pf["finale"], s=10, color=GREY, alpha=0.45, linewidths=0)
    ax.plot([1, 10], [1, 10], color=INK_2, linewidth=1, linestyle="--")
    ax.text(4.1, 3.75, "dashed line: finale = pilot", color=INK_2, fontsize=8.5)
    lab = pf[pf["primaryTitle"].isin(LABEL_SHOWS) & (pf["numVotes"] >= label_min_votes)]
    lab = lab.sort_values("numVotes", ascending=False).drop_duplicates("primaryTitle")
    for _, r in lab.iterrows():
        c = RED if r["finale_minus_pilot"] < 0 else BLUE
        ax.scatter(r["pilot"], r["finale"], s=40, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(r["primaryTitle"], (r["pilot"], r["finale"]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color=INK)
    share = (pf["finale"] > pf["pilot"]).mean()
    ax.set_xlim(4, 10)
    ax.set_ylim(1.5, 10.2)
    ax.set_xlabel("Pilot episode rating")
    ax.set_ylabel("Final episode rating")
    ax.set_title(f"Pilot vs finale for {len(pf):,} finished shows\n"
                 f"{share:.0%} of finales out-rate their pilot", loc="left")
    return fig


# ---------------------------------------------------------------------------
# 4. where in a season the best episodes sit
# ---------------------------------------------------------------------------
def episode_position(series, episodes, min_votes=5000, min_season_len=6):
    keep = series.loc[series["numVotes"] >= min_votes, "tconst"]
    e = episodes[episodes["parentTconst"].isin(keep)].copy()
    grp = e.groupby(["parentTconst", "seasonNumber"])
    e["season_len"] = grp["episodeNumber"].transform("max")
    e = e[e["season_len"] >= min_season_len]
    e["vs_season_avg"] = e["averageRating"] - e.groupby(["parentTconst", "seasonNumber"])["averageRating"].transform("mean")
    e["position"] = np.select(
        [e["episodeNumber"] == 1, e["episodeNumber"] == e["season_len"], e["episodeNumber"] == e["season_len"] - 1],
        ["Premiere", "Finale", "Penultimate"], "Middle episodes")
    order = ["Premiere", "Middle episodes", "Penultimate", "Finale"]
    t = e.groupby("position")["vs_season_avg"].agg(["mean", "count"]).reindex(order)
    t["share_above_avg"] = e.assign(above=e["vs_season_avg"] > 0).groupby("position")["above"].mean().reindex(order)
    return t.round(3)


def plot_episode_position(t):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [BLUE if v > 0 else GREY for v in t["mean"]]
    bars = ax.bar(t.index, t["mean"], color=colors, width=0.55)
    ax.axhline(0, color=INK_2, linewidth=1)
    for b, v in zip(bars, t["mean"]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.008 if v >= 0 else -0.008), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    ax.set_ylabel("Rating vs that season's average")
    ax.set_ylim(min(0, t["mean"].min()) - 0.04, t["mean"].max() + 0.04)
    ax.set_title("Season finales are the best-rated episodes of a season", loc="left")
    ax.grid(axis="x", visible=False)
    return fig


# ---------------------------------------------------------------------------
# 5. per-show views
# ---------------------------------------------------------------------------
def episode_grid(episodes, tconst):
    e = episodes[episodes["parentTconst"] == tconst]
    return e.pivot_table(index="seasonNumber", columns="episodeNumber", values="averageRating")


def plot_show_heatmaps(series, episodes, titles, ncols=2):
    nrows = int(np.ceil(len(titles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.3 * nrows), layout="constrained")
    for ax, title in zip(np.ravel(axes), titles):
        grid = episode_grid(episodes, find(series, title))
        sns.heatmap(grid, ax=ax, cmap=RATING_CMAP, vmin=4, vmax=10, annot=grid.shape[1] <= 21,
                    fmt=".1f", annot_kws={"fontsize": 7}, cbar=False, linewidths=1, linecolor=SURFACE)
        ax.grid(False)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Season")
        ax.tick_params(labelsize=8, rotation=0)
    for ax in np.ravel(axes)[len(titles):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=RATING_CMAP, norm=plt.Normalize(4, 10))
    fig.colorbar(sm, ax=list(np.ravel(axes)), shrink=0.6, pad=0.01, label="Episode rating")
    fig.suptitle("Episode ratings, season by season", x=0.01, ha="left", fontweight="bold", fontsize=13)
    return fig


def plot_long_runners(series, episodes, titles, window=15):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    for title, color in zip(titles, colors):
        e = episodes[episodes["parentTconst"] == find(series, title)] \
            .sort_values(["seasonNumber", "episodeNumber"])
        x = e["seasonNumber"] + (e["episodeNumber"] - 1) / e.groupby("seasonNumber")["episodeNumber"].transform("max")
        ax.scatter(x, e["averageRating"], s=5, color=color, alpha=0.25, linewidths=0)
        ax.plot(x, e["averageRating"].rolling(window, center=True, min_periods=5).mean(),
                color=color, linewidth=2, label=title)
    ax.set_xlabel("Season")
    ax.set_ylabel("Episode rating")
    ax.set_title(f"Long-running cartoons ({window}-episode rolling average)", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    return fig
