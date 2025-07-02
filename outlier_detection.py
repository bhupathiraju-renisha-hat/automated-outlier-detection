import os
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pandarallel import pandarallel
import json

warnings.filterwarnings("ignore", message="Duplicate values are leading to incorrect results")
pandarallel.initialize(progress_bar=True)

def rolling_window_outlier(df, col, window, k=3):
    rolling_mean = df[col].rolling(window=window).mean()
    rolling_std = df[col].rolling(window=window).std()
    outliers = abs(df[col] - rolling_mean) > (k * rolling_std)
    return outliers.fillna(False)

def iqr_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[col] < lower_bound) | (df[col] > upper_bound)

def zscore_outlier(df, col, threshold=3):
    z_scores = stats.zscore(df[col], nan_policy='omit')
    return pd.Series(np.abs(z_scores) > threshold, index=df.index)

def isolation_forest_outlier(df, col, contamination=0.05):
    if len(df) > 5:
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(df[[col]])
        return pd.Series(preds == -1, index=df.index)
    else:
        return pd.Series([False]*len(df), index=df.index)

def local_outlier_factor_outlier(df, col, contamination=0.05):
    n_samples = len(df)
    if n_samples > 1:
        unique_vals = df[col].nunique()
        dup_ratio = (n_samples - unique_vals) / n_samples
        n_neighbors = min(40 if dup_ratio > 0.5 else 20, n_samples - 1)
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds = model.fit_predict(df[[col]])
        return pd.Series(preds == -1, index=df.index)
    else:
        return pd.Series([False]*len(df), index=df.index)

def one_class_svm_outlier(df, col, nu=0.05, gamma=0.1):
    if len(df) > 5:
        model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        preds = model.fit_predict(np.array(df[col]).reshape(-1, 1))
        return pd.Series(preds == -1, index=df.index)
    else:
        return pd.Series([False]*len(df), index=df.index)

def dbscan_outlier(df, col, eps=0.5, min_samples=5):
    if len(df) > 5:
        scaled_data = StandardScaler().fit_transform(df[[col]])
        db = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = db.fit_predict(scaled_data)
        return pd.Series(clusters == -1, index=df.index)
    else:
        return pd.Series([False]*len(df), index=df.index)

def process_group(group, datetime_field, key_fields, value_field):
    select_schemaname = group[key_fields[0]].iloc[0]
    select_tablename = group[key_fields[1]].iloc[0]

    group = group.copy()
    group["timestamp"] = pd.to_datetime(group[datetime_field]).dt.floor("D")
    daily_data = group.groupby("timestamp").agg({value_field: "max"}).reset_index()

    daily_data[key_fields[0]] = select_schemaname
    daily_data[key_fields[1]] = select_tablename

    if len(daily_data) < 2 or daily_data[value_field].nunique() <= 1:
        for method in [
            "outlier_rollingwindow_7d",
            "outlier_rollingwindow_30d",
            "outlier_iqr",
            "outlier_zscore",
            "outlier_mlapproaches",
            "outlier_m5",
            "outlier_m6",
            "outlier_m7",
        ]:
            daily_data[method] = False
        daily_data["total_outliers_detected"] = 0
        return daily_data

    daily_data["outlier_rollingwindow_7d"] = rolling_window_outlier(daily_data, value_field, window=7)
    daily_data["outlier_rollingwindow_30d"] = rolling_window_outlier(daily_data, value_field, window=30)
    daily_data["outlier_iqr"] = iqr_outlier(daily_data, value_field)
    daily_data["outlier_zscore"] = zscore_outlier(daily_data, value_field)
    daily_data["outlier_mlapproaches"] = isolation_forest_outlier(daily_data, value_field)
    daily_data["outlier_m5"] = local_outlier_factor_outlier(daily_data, value_field)
    daily_data["outlier_m6"] = one_class_svm_outlier(daily_data, value_field)
    daily_data["outlier_m7"] = dbscan_outlier(daily_data, value_field)

    outlier_cols = [col for col in daily_data.columns if col.startswith("outlier_")]
    daily_data["total_outliers_detected"] = daily_data[outlier_cols].sum(axis=1)

    return daily_data

if __name__ == "__main__":
    with open("config.json", "r") as f:
        master_config = json.load(f)

    for dataset in master_config["datasets"]:
        print(f"Processing dataset: {dataset['name']}")
        df = pd.read_csv(dataset["input_file"])

        required_cols = set(dataset["key_fields"] + [dataset["datetime_field"]] + dataset["value_fields"])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input CSV missing required columns: {missing_cols}")

        os.makedirs(master_config["output_folder"], exist_ok=True)

        all_results = []
        for value_field in dataset["value_fields"]:
            results = df.groupby(dataset["key_fields"]).parallel_apply(
                lambda g: process_group(
                    g,
                    datetime_field=dataset["datetime_field"],
                    key_fields=dataset["key_fields"],
                    value_field=value_field,
                )
            )
            all_results.append(results.reset_index(drop=True))

        combined_df = all_results[0]
        for i, value_field in enumerate(dataset["value_fields"][1:], start=1):
            key_cols = dataset["key_fields"] + ["timestamp"]
            combined_df = combined_df.merge(
                all_results[i],
                on=key_cols,
                suffixes=("", f"_{value_field}")
            )

        output_file_base = os.path.join(master_config["output_folder"], f"{dataset['name']}_outlier_detection")

        formats = master_config.get("output_formats", ["excel"])
        if "excel" in formats:
            combined_df.to_excel(f"{output_file_base}.xlsx", index=False)
            print(f"Excel saved to: {output_file_base}.xlsx")
        if "csv" in formats:
            combined_df.to_csv(f"{output_file_base}.csv", index=False)
            print(f"CSV saved to: {output_file_base}.csv")
