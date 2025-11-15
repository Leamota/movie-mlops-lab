# monitor_drift.py
import pandas as pd
import numpy as np
from utils import load_model_and_features, make_feature_frame
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

USERS_REF = "users_ref.parquet"
USERS_CUR = "users_cur.parquet"
MOVIES = "movies.parquet"
INTERACTIONS_REF = "interactions_ref.parquet"
INTERACTIONS_CUR = "interactions_cur.parquet"

def score_batch(users_df, movies_df, model, feature_order, sample_users=500):
    # Score a subset to emulate production batches
    sampled_users = users_df.sample(n=sample_users, random_state=0)
    logs = []
    for _, u in sampled_users.iterrows():
        feats = make_feature_frame(u, movies_df)
        # Inference
        X = feats[feature_order]
        scores = model.predict_proba(X)[:, 1]
        feats["score"] = scores
        # Top-5 recommendations
        top5 = feats.nlargest(5, "score")
        logs.append(top5[["user_id", "movie_id", "score"] + feature_order])
    batch = pd.concat(logs, ignore_index=True)
    return batch

if __name__ == "__main__":
    # Load artifacts
    ref_users = pd.read_parquet(USERS_REF)
    cur_users = pd.read_parquet(USERS_CUR)
    movies = pd.read_parquet(MOVIES)
    model, feature_order = load_model_and_features()

    # Create reference and current batches of recommendation logs
    ref_batch = score_batch(ref_users, movies, model, feature_order, sample_users=400)
    cur_batch = score_batch(cur_users, movies, model, feature_order, sample_users=400)

    # Join with ground truth watched labels from interactions for analysis
    interactions_ref = pd.read_parquet(INTERACTIONS_REF)
    interactions_cur = pd.read_parquet(INTERACTIONS_CUR)

    def attach_labels(batch, interactions):
        merged = batch.merge(
            interactions[["user_id", "movie_id", "watched"]],
            on=["user_id", "movie_id"],
            how="left",
        )
        return merged

    ref_logs = attach_labels(ref_batch, interactions_ref)
    cur_logs = attach_labels(cur_batch, interactions_cur)
    ref_logs.to_parquet("logs_ref.parquet")
    cur_logs.to_parquet("logs_cur.parquet")

    # Evidently: build a data drift report on features and scores
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_logs[feature_order + ["score", "watched"]],
               current_data=cur_logs[feature_order + ["score", "watched"]])
    # Save HTML dashboard
    report.save_html("data_drift_report.html")
    print("Generated data_drift_report.html")
