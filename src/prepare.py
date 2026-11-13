"""
prepare.py - download the official IMDb datasets and build two tidy tables.

    data/processed/series.parquet    one row per TV series / mini-series
    data/processed/episodes.parquet  one row per rated episode

Usage:
    python src/prepare.py            # downloads anything missing, then processes
    python src/prepare.py --skip-download

IMDb datasets: https://developer.imdb.com/non-commercial-datasets/
(personal / non-commercial use; the raw files are not committed to this repo)
"""

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
FILES = ["title.basics", "title.episode", "title.ratings"]
URL = "https://datasets.imdbws.com/{}.tsv.gz"

SERIES_TYPES = {"tvSeries", "tvMiniSeries"}


def download():
    os.makedirs(RAW, exist_ok=True)
    for name in FILES:
        path = os.path.join(RAW, f"{name}.tsv.gz")
        if os.path.exists(path):
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(URL.format(name), path)


def read_tsv(name, usecols, chunksize=None, **kw):
    return pd.read_csv(os.path.join(RAW, f"{name}.tsv.gz"), sep="\t", na_values="\\N",
                       usecols=usecols, dtype=str, quoting=3, chunksize=chunksize, **kw)


def load_basics():
    """Keep only series and episodes - the other ~8M titles (films, shorts...) aren't needed."""
    keep = []
    for chunk in read_tsv("title.basics", ["tconst", "titleType", "primaryTitle", "originalTitle",
                                           "startYear", "endYear", "genres"], chunksize=1_000_000):
        keep.append(chunk[chunk["titleType"].isin(SERIES_TYPES | {"tvEpisode"})])
    b = pd.concat(keep, ignore_index=True)
    for c in ["startYear", "endYear"]:
        b[c] = pd.to_numeric(b[c], errors="coerce").astype("Int32")
    return b


ANIME_IDS = os.path.join(ROOT, "data", "reference", "anime_imdb_ids.csv")
WIKIDATA_QUERY = """
SELECT DISTINCT ?imdb WHERE {
  { ?item wdt:P31/wdt:P279* wd:Q63952888 . }                      # anime television series
  UNION { ?item wdt:P31/wdt:P279* wd:Q5398426 . ?item wdt:P136 wd:Q1107 . }  # TV series, genre anime
  UNION { ?item wdt:P31 wd:Q1259759 . }                           # original video animation
  ?item wdt:P345 ?imdb .
}"""


def anime_ids():
    """
    IMDb has no "anime" genre - anime is just tagged Animation, same as The Simpsons.
    Wikidata does know which shows are anime and stores their IMDb IDs, so the list is
    pulled from there once and cached in data/reference/ (Wikidata is CC0).
    """
    if not os.path.exists(ANIME_IDS):
        import urllib.parse
        os.makedirs(os.path.dirname(ANIME_IDS), exist_ok=True)
        req = urllib.request.Request(
            "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": WIKIDATA_QUERY}),
            headers={"Accept": "text/csv", "User-Agent": "imdb-tv-ratings-analysis/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(ANIME_IDS, "wb") as f:
            f.write(r.read())
    return set(pd.read_csv(ANIME_IDS)["imdb"])


def category(row):
    g = str(row.genres)
    if row.is_anime:
        return "Anime"
    if "Animation" in g:
        return "Western animation"
    if "Reality-TV" in g or "Game-Show" in g or "Talk-Show" in g:
        return "Reality / talk"
    if "Documentary" in g:
        return "Documentary"
    if "Comedy" in g and "Drama" not in g:
        return "Live-action comedy"
    if "Drama" in g or "Crime" in g or "Thriller" in g or "Sci-Fi" in g or "Fantasy" in g:
        return "Live-action drama"
    return "Other"


def main(skip_download=False):
    if not skip_download:
        download()
    os.makedirs(OUT, exist_ok=True)

    print("reading basics ...")
    basics = load_basics()
    ratings = read_tsv("title.ratings", None)
    ratings["averageRating"] = ratings["averageRating"].astype(float)
    ratings["numVotes"] = ratings["numVotes"].astype(int)

    print("reading episodes ...")
    ep = read_tsv("title.episode", None)
    ep["seasonNumber"] = pd.to_numeric(ep["seasonNumber"], errors="coerce").astype("Int32")
    ep["episodeNumber"] = pd.to_numeric(ep["episodeNumber"], errors="coerce").astype("Int32")

    # ---- episodes: only rated ones with a season + episode number ----
    episodes = (ep.merge(ratings, on="tconst")
                  .merge(basics[["tconst", "primaryTitle", "startYear"]], on="tconst", how="left")
                  .rename(columns={"primaryTitle": "episodeTitle", "startYear": "airYear"}))
    episodes = episodes.dropna(subset=["seasonNumber", "episodeNumber"])
    episodes = episodes[(episodes["seasonNumber"] > 0) & (episodes["episodeNumber"] > 0)]

    # ---- series ----
    series = basics[basics["titleType"].isin(SERIES_TYPES)].merge(ratings, on="tconst", how="inner")
    series["is_anime"] = series["tconst"].isin(anime_ids()) & series["genres"].str.contains("Animation", na=False)
    series["category"] = series.apply(category, axis=1)

    ep_stats = episodes.groupby("parentTconst").agg(
        rated_episodes=("tconst", "count"),
        seasons=("seasonNumber", "nunique"),
        episode_votes=("numVotes", "sum"),
    )
    series = series.merge(ep_stats, left_on="tconst", right_index=True, how="left")
    series[["rated_episodes", "seasons", "episode_votes"]] = series[["rated_episodes", "seasons", "episode_votes"]].fillna(0).astype(int)

    episodes = episodes[episodes["parentTconst"].isin(series["tconst"])]
    episodes = episodes.merge(series[["tconst", "primaryTitle", "category"]].rename(
        columns={"tconst": "parentTconst", "primaryTitle": "series"}), on="parentTconst")

    series.to_parquet(os.path.join(OUT, "series.parquet"), index=False)
    episodes.to_parquet(os.path.join(OUT, "episodes.parquet"), index=False)
    print(f"series:   {len(series):,} rated series ({series['is_anime'].sum():,} anime)")
    print(f"episodes: {len(episodes):,} rated episodes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true")
    main(p.parse_args().skip_download)
