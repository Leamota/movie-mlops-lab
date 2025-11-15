# train_baseline.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

REF_PATH = "interactions_ref.parquet"

def build_pipeline(cat_cols, num_cols):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

if __name__ == "__main__":
    df = pd.read_parquet(REF_PATH)

    cat_cols = ["region", "device", "age_bucket"] + [c for c in df.columns if c.startswith("genre_")]
    num_cols = ["duration_min", "release_year"] + [c for c in df.columns if c.startswith("aff_")]

    X = df[cat_cols + num_cols]
    y = df["watched"]

    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X, y)

    # Quick offline metrics
    y_pred = pipe.predict_proba(X)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y, y_pred):.3f}")
    print(f"PR-AUC: {average_precision_score(y, y_pred):.3f}")

    joblib.dump(pipe, "watch_model.joblib")
    pd.DataFrame({"feature": cat_cols + num_cols}).to_csv("feature_order.csv", index=False)
    print("Model saved.")
