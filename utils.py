import pandas as pd
import plotly.express as px
import streamlit as st

from src.metrics import detect_outliers


def render_engineer_view(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No performance data loaded.")
        return
    st.subheader("Diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing cells", int(df.isna().sum().sum()))
    c2.metric("Duplicate rows", int(df.duplicated().sum()))
    c3.metric("Numeric columns", len(df.select_dtypes(include="number").columns))

    outliers = detect_outliers(df)
    if not outliers.empty:
        st.warning("Potential outliers detected.")
        st.dataframe(outliers, use_container_width=True)
    else:
        st.success("No IQR outliers detected in numeric columns.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col = st.selectbox("Distribution metric", numeric_cols)
        st.plotly_chart(px.box(df, x="model_name", y=col, color="profile_id", title=f"Distribution of {col}"), use_container_width=True)

    columns = st.multiselect("Columns to inspect", df.columns.tolist(), default=df.columns.tolist()[:12])
    st.dataframe(df[columns] if columns else df, use_container_width=True)
    st.download_button("Download parsed CSV", df.to_csv(index=False), "parsed_perf_data.csv", "text/csv")
