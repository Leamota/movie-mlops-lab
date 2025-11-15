# quick_check.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

ref = pd.read_parquet("logs_ref.parquet")
cur = pd.read_parquet("logs_cur.parquet")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
summary = report.as_dict()

drifted = summary["metrics"][0]["result"]["data_drift"]["dataset_drift"]
pvalue = summary["metrics"][0]["result"]["data_drift"]["drift_share"]
print(f"Dataset drift: {drifted}, drift share: {pvalue:.2f}")

if drifted or pvalue > 0.3:
    print("ALERT: Significant drift detected. Schedule retraining.")
