from pathlib import Path
import streamlit as st

from app.shared_ui import page_header
from app.task1_dashboard import render_task1
from app.task2_dashboard import render_task2_pruning, render_mmmu_probe, render_docs

st.set_page_config(page_title="Cerebras AI Engineer Challenge", layout="wide")

PAGES = {
    "Task 1 — Performance Explorer": render_task1,
    "Task 2 — Pruning Results": render_task2_pruning,
    "Task 2 — MMMU Probe": render_mmmu_probe,
    "Documentation / Methodology": render_docs,
}

with st.sidebar:
    st.title("Cerebras Challenge")
    choice = st.radio("Navigate", list(PAGES.keys()))
    st.caption("Submission-ready Streamlit interface for Task 1 and Task 2.")

page_header(choice)
PAGES[choice](Path(__file__).parent)
