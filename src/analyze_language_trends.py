#!/usr/bin/env python3
"""Week 4: temporal NLP analysis of HuffPost News Category Dataset.

Downloads the public HuffPost News Category Dataset mirror and analyzes
vocabulary, tracked phrases, LDA topics, TF-IDF clusters, and category mix by year.

Example:
    pip install -r requirements.txt
    python src/analyze_language_trends.py --out_dir outputs

A local JSON/JSONL/CSV file can also be passed with --input_file.
The full dataset is used for annual frequency/category analysis. Topic and
cluster models are fitted on reproducible samples for practical runtime.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from scipy.stats import linregress, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

DATASET_ID = "gohelrakesh/news-category"
TRACKED_PHRASES = [
    "artificial intelligence", "machine learning", "deep learning",
    "social media", "climate change", "brexit", "covid", "coronavirus",
    "cryptocurrency", "bitcoin", "tiktok", "instagram", "podcast",
    "streaming", "remote work", "selfie", "metaverse", "electric vehicle",
]
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze news language trends over time")
    parser.add_argument("--out_dir", default="outputs", help="Directory for CSV, JSON, and chart outputs")
    parser.add_argument("--dataset_id", default=DATASET_ID, help="Hugging Face dataset identifier")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--input_file", default=None, help="Optional local CSV, JSON, or JSONL file")
    parser.add_argument("--max_documents", type=int, default=None, help="Optional cap for quick testing; omit for full dataset")
    parser.add_argument("--topic_sample_size", type=int, default=40000, help="Documents used to fit LDA")
    parser.add_argument("--cluster_sample_size", type=int, default=30000, help="Documents used to fit MiniBatchKMeans")
    parser.add_argument("--n_topics", type=int, default=8, help="Number of LDA topics")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of TF-IDF clusters")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--min_year_docs", type=int, default=100, help="Minimum annual documents for trend tests")
    return parser.parse_args()


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            raise FileNotFoundError(f"Local data file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True)
        elif suffix == ".json":
            frame = pd.read_json(path, lines=False)
        else:
            raise ValueError("--input_file must be .csv, .json, or .jsonl")
    else:
        if load_dataset is None:
            raise RuntimeError("The datasets package is required for Hugging Face downloads. Run: pip install -r requirements.txt")
        print(f"Loading {args.dataset_id} ({args.split}) from Hugging Face Datasets...")
        dataset = load_dataset(args.dataset_id, split=args.split)
        frame = dataset.to_pandas()
    required = {"headline", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    if "short_description" not in frame.columns:
        frame["short_description"] = ""
    if "category" not in frame.columns:
        frame["category"] = "Unknown"
    if args.max_documents:
        frame = frame.sample(n=min(args.max_documents, len(frame)), random_state=args.random_state)
    return frame.reset_index(drop=True)


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data["year"] = data["date"].dt.year
    data["headline"] = data["headline"].fillna("").astype(str)
    data["short_description"] = data["short_description"].fillna("").astype(str)
    data["category"] = data["category"].fillna("Unknown").astype(str)
    data["text"] = (data["headline"].str.strip() + ". " + data["short_description"].str.strip()).str.strip()
    data["clean_text"] = data["text"].map(clean_text)
    data["headline_tokens"] = data["headline"].map(lambda x: re.findall(r"(?u)\b\w+(?:['’]\w+)?\b", x.lower()))
    data["headline_word_count"] = data["headline_tokens"].map(len)
    data = data.dropna(subset=["date", "year"])
    data = data.loc[data["clean_text"].str.len() > 0].copy()
    data["year"] = data["year"].astype(int)
    return data.reset_index(drop=True)


def clean_text(value: str) -> str:
    value = URL_RE.sub(" URLTOKEN ", str(value))
    value = HTML_RE.sub(" ", value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9'’#@%$+\-\s]", " ", value)
    return SPACE_RE.sub(" ", value).strip()


def save_yearly_descriptives(data: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    yearly = data.groupby("year").agg(
        documents=("text", "size"),
        mean_headline_words=("headline_word_count", "mean"),
        median_headline_words=("headline_word_count", "median"),
        categories=("category", "nunique"),
    ).reset_index()
    yearly.to_csv(out_dir / "yearly_dataset_profile.csv", index=False)
    category_top = data["category"].value_counts().head(12).index
    subset = data[data["category"].isin(category_top)]
    counts = subset.groupby(["year", "category"]).size().rename("count").reset_index()
    totals = data.groupby("year").size().rename("year_total").reset_index()
    counts = counts.merge(totals, on="year")
    counts["share_percent"] = counts["count"] / counts["year_total"] * 100
    counts.to_csv(out_dir / "category_share_by_year.csv", index=False)
    return yearly


def frequency_analysis(data: pd.DataFrame, out_dir: Path) -> tuple[CountVectorizer, Any]:
    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=5,
        max_df=0.85, max_features=20000, token_pattern=r"(?u)\b\w[\w'’#@+-]*\b",
    )
    matrix = vectorizer.fit_transform(data["clean_text"].tolist())
    features = vectorizer.get_feature_names_out()
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    rows = pd.DataFrame({"ngram": features, "count": totals.astype(int)})
    rows["type"] = rows["ngram"].map(lambda s: "bigram" if " " in s else "unigram")
    rows.sort_values("count", ascending=False).to_csv(out_dir / "top_ngrams.csv", index=False)
    return vectorizer, matrix


def tracked_term_analysis(data: pd.DataFrame, out_dir: Path, min_year_docs: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    text_series = data["clean_text"]
    yearly_counts = data.groupby("year").size()
    for phrase in TRACKED_PHRASES:
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", flags=re.I)
        matching = text_series.map(lambda text: bool(pattern.search(text)))
        counts = data.assign(_match=matching).groupby("year")["_match"].sum()
        for year, n_docs in yearly_counts.items():
            count = int(counts.get(year, 0))
            rate = count / int(n_docs) * 1000 if n_docs else 0.0
            rows.append({"phrase": phrase, "year": int(year), "documents": int(n_docs), "document_count_with_phrase": count, "rate_per_1000_documents": rate})
    trends = pd.DataFrame(rows)
    trends.to_csv(out_dir / "tracked_phrase_trends.csv", index=False)
    stat_rows = []
    for phrase, group in trends.groupby("phrase"):
        valid = group[group["documents"] >= min_year_docs]
        if len(valid) >= 3:
            slope = linregress(valid["year"], valid["rate_per_1000_documents"])
            if valid["rate_per_1000_documents"].nunique() > 1:
                rho, pval = spearmanr(valid["year"], valid["rate_per_1000_documents"])
                rho_value, p_value = float(rho), float(pval)
            else:
                rho_value, p_value = np.nan, np.nan
            stat_rows.append({"phrase": phrase, "years_included": len(valid), "slope_rate_per_year": float(slope.slope), "spearman_rho": rho_value, "spearman_p_value": p_value, "direction": "increasing" if slope.slope > 0 else "decreasing" if slope.slope < 0 else "flat"})
        else:
            stat_rows.append({"phrase": phrase, "years_included": len(valid), "slope_rate_per_year": np.nan, "spearman_rho": np.nan, "spearman_p_value": np.nan, "direction": "insufficient annual volume"})
    pd.DataFrame(stat_rows).to_csv(out_dir / "tracked_phrase_trend_statistics.csv", index=False)
    return trends


def fit_topic_model(data: pd.DataFrame, count_matrix: Any, vectorizer: CountVectorizer, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_state)
    n = len(data)
    sample_n = min(args.topic_sample_size, n)
    sample_idx = np.sort(rng.choice(n, size=sample_n, replace=False))
    sample_matrix = count_matrix[sample_idx]
    lda = LatentDirichletAllocation(
        n_components=args.n_topics, max_iter=8, learning_method="online",
        learning_decay=0.7, batch_size=512, evaluate_every=-1,
        random_state=args.random_state, n_jobs=-1,
    )
    lda.fit(sample_matrix)
    terms = vectorizer.get_feature_names_out()
    term_rows = []
    for topic_id, weights in enumerate(lda.components_):
        top_idx = np.argsort(weights)[-12:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            term_rows.append({"topic_id": topic_id, "rank": rank, "term": terms[idx], "weight": float(weights[idx])})
    pd.DataFrame(term_rows).to_csv(out_dir / "lda_topic_terms.csv", index=False)

    yearly_values = sorted(data["year"].unique())
    year_topic_totals: dict[int, np.ndarray] = {int(y): np.zeros(args.n_topics, dtype=float) for y in yearly_values}
    year_counts: Counter[int] = Counter()
    for start in range(0, n, 5000):
        end = min(start + 5000, n)
        dist = lda.transform(count_matrix[start:end])
        years = data["year"].iloc[start:end].to_numpy()
        for year in np.unique(years):
            mask = years == year
            year_topic_totals[int(year)] += dist[mask].sum(axis=0)
            year_counts[int(year)] += int(mask.sum())
    topic_rows = []
    for year in yearly_values:
        denominator = year_counts[int(year)] or 1
        average = year_topic_totals[int(year)] / denominator
        for topic_id, value in enumerate(average):
            topic_rows.append({"year": int(year), "topic_id": topic_id, "mean_topic_probability": float(value)})
    topic_by_year = pd.DataFrame(topic_rows)
    topic_by_year.to_csv(out_dir / "lda_topic_prevalence_by_year.csv", index=False)
    metadata = {"topics": int(args.n_topics), "fitted_on_documents": int(sample_n), "training_perplexity_sample": float(lda.perplexity(sample_matrix)), "random_state": int(args.random_state), "note": "LDA topics are unsupervised and require human interpretation; perplexity is a model-fit diagnostic, not a direct measure of semantic quality."}
    (out_dir / "lda_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return topic_by_year


def fit_clusters(data: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_state + 1)
    n = len(data)
    sample_n = min(args.cluster_sample_size, n)
    sample_idx = np.sort(rng.choice(n, size=sample_n, replace=False))
    sample_texts = data["clean_text"].iloc[sample_idx].tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=4,
        max_df=0.85, max_features=15000, sublinear_tf=True,
    )
    sample_tfidf = vectorizer.fit_transform(sample_texts)
    clusterer = MiniBatchKMeans(
        n_clusters=args.n_clusters, random_state=args.random_state,
        batch_size=1024, n_init=3, max_iter=100,
    )
    clusterer.fit(sample_tfidf)
    cluster_ids = np.empty(n, dtype=int)
    for start in range(0, n, 5000):
        end = min(start + 5000, n)
        cluster_ids[start:end] = clusterer.predict(vectorizer.transform(data["clean_text"].iloc[start:end].tolist()))
    data_clusters = data[["year"]].copy()
    data_clusters["cluster_id"] = cluster_ids
    prevalence = data_clusters.groupby(["year", "cluster_id"]).size().rename("documents").reset_index()
    totals = data_clusters.groupby("year").size().rename("year_total").reset_index()
    prevalence = prevalence.merge(totals, on="year")
    prevalence["share_percent"] = prevalence["documents"] / prevalence["year_total"] * 100
    prevalence.to_csv(out_dir / "cluster_prevalence_by_year.csv", index=False)

    terms = vectorizer.get_feature_names_out()
    term_rows = []
    for cluster_id, center in enumerate(clusterer.cluster_centers_):
        top = np.argsort(center)[-12:][::-1]
        for rank, idx in enumerate(top, start=1):
            term_rows.append({"cluster_id": cluster_id, "rank": rank, "term": terms[idx], "weight": float(center[idx])})
    pd.DataFrame(term_rows).to_csv(out_dir / "cluster_top_terms.csv", index=False)
    sil_n = min(2000, sample_n)
    sil_idx = rng.choice(sample_n, size=sil_n, replace=False)
    sil_score = None
    if args.n_clusters > 1 and sil_n > args.n_clusters:
        try:
            sil_labels = clusterer.predict(sample_tfidf[sil_idx])
            if len(np.unique(sil_labels)) > 1:
                sil_score = float(silhouette_score(sample_tfidf[sil_idx], sil_labels, metric="cosine"))
        except Exception:
            sil_score = None
    (out_dir / "cluster_metadata.json").write_text(json.dumps({"clusters": int(args.n_clusters), "fitted_on_documents": int(sample_n), "silhouette_sample_size": int(sil_n), "silhouette_cosine_sample": sil_score, "random_state": int(args.random_state), "note": "Clusters are descriptive groupings; top terms must be reviewed before assigning human-readable labels."}, indent=2), encoding="utf-8")
    return prevalence


def make_charts(data: pd.DataFrame, yearly: pd.DataFrame, trends: pd.DataFrame, topics: pd.DataFrame, clusters: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(yearly["year"].astype(str), yearly["documents"])
    ax.set_title("HuffPost archive coverage by year")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of articles")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(out_dir / "01_documents_by_year.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    overall = data["category"].value_counts().head(10).index
    cat = data[data["category"].isin(overall)].groupby(["year", "category"]).size().rename("count").reset_index()
    totals = data.groupby("year").size().rename("total").reset_index()
    cat = cat.merge(totals, on="year"); cat["share"] = cat["count"] / cat["total"] * 100
    pivot = cat.pivot(index="year", columns="category", values="share").fillna(0)
    pivot.plot(ax=ax, marker="o", linewidth=1.4)
    ax.set_title("Share of the top 10 news categories by year")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Share of yearly articles (%)")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "02_category_share_by_year.png"); plt.close(fig)

    top_phrases = trends.groupby("phrase")["rate_per_1000_documents"].max().nlargest(8).index
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for phrase in top_phrases:
        g = trends[trends["phrase"] == phrase]
        ax.plot(g["year"], g["rate_per_1000_documents"], marker="o", linewidth=1.5, label=phrase)
    ax.set_title("Tracked phrase frequency over time (per 1,000 articles)")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Articles mentioning phrase per 1,000")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "03_tracked_phrase_trends.png"); plt.close(fig)

    topic_pivot = topics.pivot(index="year", columns="topic_id", values="mean_topic_probability").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    topic_pivot.plot(ax=ax, marker="o", linewidth=1.5)
    ax.set_title("Mean LDA topic probability by publication year")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Mean topic probability")
    ax.legend(title="Topic", ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "04_lda_topic_prevalence.png"); plt.close(fig)

    cluster_pivot = clusters.pivot(index="year", columns="cluster_id", values="share_percent").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cluster_pivot.plot(ax=ax, marker="o", linewidth=1.4)
    ax.set_title("TF-IDF cluster share by publication year")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Share of yearly articles (%)")
    ax.legend(title="Cluster", ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "05_cluster_share_by_year.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(yearly["year"], yearly["mean_headline_words"], marker="o", linewidth=1.8)
    ax.set_title("Average headline length by year")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Mean words per headline")
    fig.tight_layout(); fig.savefig(out_dir / "06_headline_length_by_year.png"); plt.close(fig)

    freq_path = out_dir / "top_ngrams.csv"
    if freq_path.exists():
        freq = pd.read_csv(freq_path).head(18).sort_values("count")
        fig, ax = plt.subplots(figsize=(8.5, 5.3))
        ax.barh(freq["ngram"], freq["count"])
        ax.set_title("Most frequent cleaned unigrams and bigrams")
        ax.set_xlabel("Corpus count")
        fig.tight_layout(); fig.savefig(out_dir / "07_top_ngrams.png"); plt.close(fig)
    plt.close("all")


def main() -> None:
    args = parse_args()
    if args.n_topics < 2 or args.n_clusters < 2:
        raise SystemExit("Use at least two topics and two clusters.")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_frame(args)
    data = preprocess(raw)
    if data.empty:
        raise ValueError("No usable records remain after date and text validation.")
    data.to_csv(out_dir / "cleaned_news_metadata.csv", index=False)

    yearly = save_yearly_descriptives(data, out_dir)
    count_vectorizer, count_matrix = frequency_analysis(data, out_dir)
    trends = tracked_term_analysis(data, out_dir, args.min_year_docs)
    topics = fit_topic_model(data, count_matrix, count_vectorizer, args, out_dir)
    clusters = fit_clusters(data, args, out_dir)
    make_charts(data, yearly, trends, topics, clusters, out_dir)

    metadata = {
        "project": "Advanced Analysis and Reporting of Language Trends using NLP Techniques",
        "dataset": args.dataset_id if not args.input_file else str(args.input_file),
        "loaded_records": int(len(raw)),
        "analyzed_records_after_cleaning": int(len(data)),
        "first_publication_date": str(data["date"].min()),
        "last_publication_date": str(data["date"].max()),
        "years": [int(data["year"].min()), int(data["year"].max())],
        "categories": int(data["category"].nunique()),
        "n_topics": int(args.n_topics),
        "n_clusters": int(args.n_clusters),
        "random_state": int(args.random_state),
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "limitations": [
            "Dataset archive volume is substantially denser before May 2018 than after; normalize by annual document count and interpret later years cautiously.",
            "Topic and cluster labels require human interpretation; unsupervised groups are not ground-truth categories.",
            "HuffPost is a single publisher and does not represent all English-language news or general language use.",
        ],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"\nOutputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
