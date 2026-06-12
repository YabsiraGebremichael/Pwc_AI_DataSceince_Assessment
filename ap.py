import glob
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="GameVault Acquisition Dashboard", layout="wide")

# ── Data pipeline ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and processing data…")
def load_and_process():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the current directory.")
    df = pd.read_csv(csv_files[0])

    # Cleaning
    clean = df.copy()
    clean = clean.dropna(subset=["title"])
    clean["review"] = clean["review"].fillna("")
    for col in ["funny", "helpful", "hour_played"]:
        clean[col] = clean[col].fillna(0)
    clean["date_posted"] = pd.to_datetime(clean["date_posted"], errors="coerce")
    clean["recommended"] = (clean["recommendation"] == "Recommended").astype(int)
    clean = clean.drop_duplicates()
    p99 = clean["hour_played"].quantile(0.99)
    clean["hour_played"] = clean["hour_played"].clip(upper=p99)

    # Game-level aggregation
    dataset_median_hours = clean["hour_played"].median()
    game = (
        clean.groupby("title")
        .agg(
            total_reviews=("recommended", "count"),
            recommendation_rate=("recommended", "mean"),
            avg_hours_played=("hour_played", "mean"),
            median_hours_played=("hour_played", "median"),
            helpful_votes_total=("helpful", "sum"),
            first_review=("date_posted", "min"),
            last_review=("date_posted", "max"),
        )
        .reset_index()
    )
    game["review_period_days"] = (game["last_review"] - game["first_review"]).dt.days
    game["reviews_per_month"] = (
        game["total_reviews"] / game["review_period_days"].replace(0, 1) * 30
    )
    high_eng = (
        clean.assign(is_high_eng=(clean["hour_played"] > dataset_median_hours).astype(int))
        .groupby("title")["is_high_eng"]
        .mean()
        .rename("high_engagement_rate")
    )
    game = game.merge(high_eng, on="title")
    game = game.drop(columns=["first_review", "last_review"])

    # Sentiment
    clean["sentiment_polarity"] = clean["review"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity
    )
    sentiment = (
        clean.groupby("title")["sentiment_polarity"].mean().rename("sentiment_score")
    )
    game = game.merge(sentiment, on="title")

    # Acquisition score
    SCORE_FEATURES = [
        "recommendation_rate",
        "avg_hours_played",
        "total_reviews",
        "sentiment_score",
        "review_period_days",
    ]
    scaled = game[SCORE_FEATURES].copy()
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    game["acquisition_score"] = scaled.mean(axis=1) * 100

    # Target variable
    hours_median = game["avg_hours_played"].median()
    reviews_p25 = game["total_reviews"].quantile(0.25)
    game["strong_candidate"] = (
        (game["recommendation_rate"] >= 0.75)
        & (game["avg_hours_played"] >= hours_median)
        & (game["total_reviews"] >= reviews_p25)
    ).astype(int)

    # Random Forest
    FEATURES = [
        "recommendation_rate",
        "avg_hours_played",
        "total_reviews",
        "sentiment_score",
        "review_period_days",
        "reviews_per_month",
        "high_engagement_rate",
    ]
    X = game[FEATURES]
    y = game["strong_candidate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    game["predicted_candidate"] = rf.predict(X)
    game["predicted_proba"] = rf.predict_proba(X)[:, 1]

    return game


game = load_and_process()

# ── Sidebar navigation ────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Game Explorer", "Top Acquisition Targets"],
)

# ── Page 1: Overview ──────────────────────────────────────────────────────

if page == "Overview":
    st.title("GameVault Publishing — Acquisition Dashboard")

    st.markdown(
        "GameVault Publishing is building a subscription game catalogue and needs a "
        "data-driven way to identify which titles are worth licensing — balancing player "
        "satisfaction, depth of engagement, and long-term appeal against acquisition cost. "
        "This dashboard analyses Steam review data to rank candidates using a composite "
        "acquisition score and a trained Random Forest classifier, helping the team focus "
        "negotiation effort on the titles most likely to drive subscriber retention."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Games Analysed", len(game))
    col2.metric(
        "Avg Recommendation Rate",
        f"{game['recommendation_rate'].mean():.1%}",
    )
    col3.metric(
        "Avg Acquisition Score",
        f"{game['acquisition_score'].mean():.1f} / 100",
    )

# ── Page 2: Game Explorer ─────────────────────────────────────────────────

elif page == "Game Explorer":
    st.title("Game Explorer")

    min_score = st.slider(
        "Minimum Acquisition Score",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )
    min_reviews = st.slider(
        "Minimum Total Reviews",
        min_value=0,
        max_value=int(game["total_reviews"].max()),
        value=0,
        step=10,
    )

    display_cols = [
        "title",
        "acquisition_score",
        "recommendation_rate",
        "avg_hours_played",
        "sentiment_score",
        "total_reviews",
        "review_period_days",
        "reviews_per_month",
        "high_engagement_rate",
        "strong_candidate",
        "predicted_candidate",
        "predicted_proba",
    ]

    filtered = (
        game[
            (game["acquisition_score"] >= min_score)
            & (game["total_reviews"] >= min_reviews)
        ][display_cols]
        .sort_values("acquisition_score", ascending=False)
        .reset_index(drop=True)
    )

    st.caption(f"{len(filtered)} game(s) match the current filters.")
    st.dataframe(filtered, use_container_width=True)

# ── Page 3: Top Acquisition Targets ──────────────────────────────────────

elif page == "Top Acquisition Targets":
    st.title("Top Acquisition Targets")

    top10 = game.nlargest(10, "acquisition_score").sort_values("acquisition_score")

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10["title"], top10["acquisition_score"], color="#4C78A8")
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    ax.set_xlabel("Acquisition Score (0–100)")
    ax.set_title("Top 10 Games by Acquisition Score")
    ax.set_xlim(0, top10["acquisition_score"].max() * 1.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    table_cols = [
        "title",
        "acquisition_score",
        "recommendation_rate",
        "avg_hours_played",
        "sentiment_score",
        "total_reviews",
    ]
    top10_table = (
        game.nlargest(10, "acquisition_score")[table_cols]
        .reset_index(drop=True)
    )
    top10_table.index += 1
    st.dataframe(top10_table, use_container_width=True)

    st.info(
        "Games that appear in both the acquisition score ranking above **and** the Random "
        "Forest predictions (`predicted_candidate = 1`) represent the strongest candidates. "
        "Agreement between the rule-based score and the data-driven model provides the "
        "highest confidence that a title is worth prioritising in licensing negotiations."
    )
