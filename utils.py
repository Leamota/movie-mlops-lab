# utils.py
import numpy as np
import pandas as pd
import joblib

def load_model_and_features():
    model = joblib.load("watch_model.joblib")
    feature_order = pd.read_csv("feature_order.csv")["feature"].tolist()

    return model, feature_order

def make_feature_frame(users_row, movies_df):
    # Build candidate features per user across movies
    rows = []
    for _, m in movies_df.iterrows():
        row = {
            "user_id": users_row["user_id"],
            "movie_id": m["movie_id"],
            "region": users_row["region"],
            "device": users_row["device"],
            "age_bucket": users_row["age_bucket"],
            "duration_min": m["duration_min"],
            "release_year": m["release_year"],
        }
        # genres and affinities
        for c in [c for c in movies_df.columns if c.startswith("genre_")]:
            row[c] = m[c]
        for c in [c for c in users_row.index if c.startswith("aff_")]:
            row[c] = users_row[c]
        rows.append(row)
    return pd.DataFrame(rows)
