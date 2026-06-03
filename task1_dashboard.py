from pathlib import Path
import pandas as pd
import streamlit as st

from app.shared_ui import metric_row
from src.parser import load_default_perf_data, load_uploaded_perf_data
from src.metrics import summarize_dataset
from src.views_customer import render_customer_view
from src.views_engineer import render_engineer_view


@st.cache_data(show_spinner=False)
def _load_defaults(root: str) -> pd.DataFrame:
    return load_default_perf_data(root)


def render_task1(project_root: Path) -> None:
    st.markdown("Performance projections are preloaded from `data/perf_data` and uploaded sweeps are merged into the same comparison dataframe.")
    uploaded = st.file_uploader("Add optional perf sweep .xlsx files", type=["xlsx"], accept_multiple_files=True)
    default_df = _load_defaults(str(project_root / "data" / "perf_data"))
    uploaded_df = load_uploaded_perf_data(uploaded)
    final_df = pd.concat([default_df, uploaded_df], ignore_index=True) if not uploaded_df.empty else default_df

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Default files preloaded", default_df.get("source_filename", pd.Series(dtype=str)).nunique())
    c2.metric("Uploaded files added", uploaded_df.get("source_filename", pd.Series(dtype=str)).nunique() if not uploaded_df.empty else 0)
    c3.metric("Total models", final_df.get("model_name", pd.Series(dtype=str)).nunique())
    c4.metric("Total profiles", final_df.get("profile_id", pd.Series(dtype=str)).nunique())
    c5.metric("Rows parsed", f"{len(final_df):,}")

    if "parse_error" in final_df.columns and final_df["parse_error"].notna().any():
        st.error("Some files had parse errors.")
        st.dataframe(final_df[final_df["parse_error"].notna()][["source_filename", "parse_error"]], use_container_width=True)

    tab1, tab2 = st.tabs(["Customer / PM View", "Internal Engineer View"])
    with tab1:
        render_customer_view(final_df)
    with tab2:
        render_engineer_view(final_df)
