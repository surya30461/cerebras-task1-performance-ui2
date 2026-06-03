import pandas as pd
import plotly.express as px
import streamlit as st

from src.metrics import add_recommendation_columns, first_existing, THROUGHPUT_CANDIDATES, TTFT_CANDIDATES, LATENCY_CANDIDATES


def render_customer_view(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No performance data loaded.")
        return
    df = add_recommendation_columns(df)
    models = st.multiselect("Models", sorted(df["model_name"].dropna().unique()), default=sorted(df["model_name"].dropna().unique())[:6])
    profiles = st.multiselect("Profiles", sorted(df["profile_id"].dropna().unique()), default=sorted(df["profile_id"].dropna().unique()))
    fdf = df[df["model_name"].isin(models) & df["profile_id"].isin(profiles)] if models and profiles else df

    best = fdf.sort_values("recommendation_score", ascending=False).head(1)
    if not best.empty:
        row = best.iloc[0]
        st.success(f"Recommended starting point: **{row.get('model_name')} / {row.get('profile_id')}** with **{row.get('go_status', 'Caution')}** status.")

    show_cols = [c for c in ["model_name", "profile_id", "input_length", "output_length", "batch_size", first_existing(fdf, THROUGHPUT_CANDIDATES), first_existing(fdf, TTFT_CANDIDATES), first_existing(fdf, LATENCY_CANDIDATES), "go_status", "recommendation_score"] if c and c in fdf.columns]
    st.subheader("Model / profile comparison")
    st.dataframe(fdf[show_cols].sort_values("recommendation_score", ascending=False), use_container_width=True)

    label = "model_name"
    tp = first_existing(fdf, THROUGHPUT_CANDIDATES)
    ttft = first_existing(fdf, TTFT_CANDIDATES)
    lat = first_existing(fdf, LATENCY_CANDIDATES)
    if tp:
        st.plotly_chart(px.bar(fdf, x=label, y=tp, color="profile_id", barmode="group", title="Throughput by model/profile"), use_container_width=True)
    if ttft:
        st.plotly_chart(px.bar(fdf, x=label, y=ttft, color="profile_id", barmode="group", title="TTFT by model/profile — lower is better"), use_container_width=True)
    if lat:
        st.plotly_chart(px.bar(fdf, x=label, y=lat, color="profile_id", barmode="group", title="Latency budget by model/profile — lower is better"), use_container_width=True)

    st.info("Customer interpretation: prioritize high throughput and generation speed while keeping TTFT and latency under the workload target. The status is a quick triage signal, not a replacement for production load testing.")
