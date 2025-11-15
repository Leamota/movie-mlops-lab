# data_simulation.py
import numpy as np
import pandas as pd

np.random.seed(42)

GENRES = ["drama", "comedy", "action", "romance", "thriller", "documentary", "animation", "anime", "scifi"]
REGIONS = ["NA", "EU", "APAC", "LATAM"]
DEVICES = ["mobile", "tv", "web"]
AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55+"]

def simulate_users(n=5000, anime_bias=0.05):
    users = pd.DataFrame({
        "user_id": np.arange(n),
        "region": np.random.choice(REGIONS, size=n, p=[0.35, 0.25, 0.25, 0.15]),
        "device": np.random.choice(DEVICES, size=n, p=[0.5, 0.35, 0.15]),
        "age_bucket": np.random.choice(AGE_BUCKETS, size=n, p=[0.25, 0.3, 0.2, 0.15, 0.1]),
    })
    # User-level genre affinity
    genre_affinity = {g: np.random.beta(2, 5, size=n) for g in GENRES}
    # Add anime bias
    genre_affinity["anime"] += anime_bias
    for g in GENRES:
        users[f"aff_{g}"] = np.clip(genre_affinity[g], 0, 1)
    return users

def simulate_catalog(m=3000):
    movies = pd.DataFrame({
        "movie_id": np.arange(m),
        "duration_min": np.random.randint(70, 140, size=m),
        "release_year": np.random.randint(1990, 2025, size=m),
    })
    # Multi-label genres
    for g in GENRES:
        movies[f"genre_{g}"] = (np.random.rand(m) < (0.18 if g != "anime" else 0.06)).astype(int)
    # Normalize at least one genre
    at_least_one = movies[[f"genre_{g}" for g in GENRES]].sum(axis=1) > 0
    missing = movies.index[~at_least_one]
    for idx in missing:
        g = np.random.choice(GENRES)
        movies.loc[idx, f"genre_{g}"] = 1
    return movies

def simulate_interactions(users, movies, n_samples=50000, drift=False):
    # Sample user-movie pairs
    u_idx = np.random.choice(users.index, size=n_samples)
    m_idx = np.random.choice(movies.index, size=n_samples)
    df = pd.DataFrame({
        "user_id": users.loc[u_idx, "user_id"].values,
        "movie_id": movies.loc[m_idx, "movie_id"].values,
        "region": users.loc[u_idx, "region"].values,
        "device": users.loc[u_idx, "device"].values,
        "age_bucket": users.loc[u_idx, "age_bucket"].values,
        "duration_min": movies.loc[m_idx, "duration_min"].values,
        "release_year": movies.loc[m_idx, "release_year"].values,
    })
    # Compute match score: user affinity dot movie genres
    for g in GENRES:
        df[f"genre_{g}"] = movies.loc[m_idx, f"genre_{g}"].values
        df[f"aff_{g}"] = users.loc[u_idx, f"aff_{g}"].values

    base_match = sum(df[f"genre_{g}"] * df[f"aff_{g}"] for g in GENRES)
    device_factor = df["device"].map({"mobile": 0.95, "tv": 1.05, "web": 0.9}).astype(float)
    region_factor = df["region"].map({"NA": 1.0, "EU": 0.98, "APAC": 1.02, "LATAM": 0.97}).astype(float)
    recency_factor = 1 + (df["release_year"] - 2000) / 100.0

    score = base_match * device_factor * region_factor * recency_factor

    # Drift scenario: surge in anime preference
    if drift:
        score += 0.5 * (df["genre_anime"] * 1)  # extra lift for anime titles

    # Convert to watch probability via sigmoid
    prob = 1 / (1 + np.exp(-3 * (score - 0.8)))
    watch = (np.random.rand(len(prob)) < prob).astype(int)

    df["watched"] = watch
    df["score"] = score
    return df

if __name__ == "__main__":
    users_ref = simulate_users(n=5000, anime_bias=0.05)
    movies = simulate_catalog(m=3000)
    interactions_ref = simulate_interactions(users_ref, movies, n_samples=60000, drift=False)

    users_cur = simulate_users(n=5000, anime_bias=0.20)  # increased anime affinity
    interactions_cur = simulate_interactions(users_cur, movies, n_samples=60000, drift=True)

    users_ref.to_parquet("users_ref.parquet")
    users_cur.to_parquet("users_cur.parquet")
    movies.to_parquet("movies.parquet")
    interactions_ref.to_parquet("interactions_ref.parquet")
    interactions_cur.to_parquet("interactions_cur.parquet")
    print("Datasets saved.")
