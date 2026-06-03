from __future__ import annotations

import pandas as pd

THROUGHPUT_CANDIDATES = ["throughput_t_s", "gen_only_throughput_t_s", "uncached_throughput_t_s", "cached_throughput_t_s"]
TTFT_CANDIDATES = ["ttft_ms", "ttft"]
LATENCY_CANDIDATES = ["max_number_of_milliseconds", "target_max_number_of_milliseconds"]
CONTEXT_CANDIDATES = ["input_length", "output_length"]


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize_dataset(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Default/uploaded rows": 0, "Models": 0, "Profiles": 0, "Files": 0}
    return {
        "Rows parsed": f"{len(df):,}",
        "Models": df.get("model_name", pd.Series(dtype=str)).nunique(),
        "Profiles": df.get("profile_id", pd.Series(dtype=str)).nunique(),
        "Files": df.get("source_filename", pd.Series(dtype=str)).nunique(),
    }


def add_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tp = first_existing(out, THROUGHPUT_CANDIDATES)
    ttft = first_existing(out, TTFT_CANDIDATES)
    if tp and out[tp].notna().any():
        out["throughput_rank"] = out[tp].rank(ascending=False, method="dense")
    if ttft and out[ttft].notna().any():
        out["ttft_rank"] = out[ttft].rank(ascending=True, method="dense")
    score = 0
    if tp:
        score = score + out[tp].rank(pct=True)
    if ttft:
        score = score + (1 - out[ttft].rank(pct=True))
    out["recommendation_score"] = score
    if isinstance(score, pd.Series) and score.max() != score.min():
        q1 = score.quantile(0.33); q2 = score.quantile(0.66)
        out["go_status"] = pd.cut(score, bins=[-float("inf"), q1, q2, float("inf")], labels=["No-Go", "Caution", "Go"])
    else:
        out["go_status"] = "Caution"
    return out


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((numeric[col] < low) | (numeric[col] > high)).sum())
        if count:
            rows.append({"column": col, "outlier_count": count, "low_threshold": low, "high_threshold": high})
    return pd.DataFrame(rows)
