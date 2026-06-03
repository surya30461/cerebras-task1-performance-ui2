import streamlit as st


def page_header(title: str) -> None:
    st.title(title)
    st.markdown("Built for the Cerebras AI Engineer — Model Quality & Performance challenge.")


def metric_row(metrics: dict) -> None:
    cols = st.columns(len(metrics) or 1)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)
