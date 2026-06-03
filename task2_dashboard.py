from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def render_task2_pruning(project_root: Path) -> None:
    results = project_root / "task2" / "results"
    summary = _read_csv(results / "summary_metrics.csv")
    comparison = _read_csv(results / "comparison.csv")
    selected = _read_csv(results / "selected_samples.csv")
    if summary.empty or comparison.empty:
        st.warning("Task 2 result files were not found yet. Run the reproducible pruning command below.")
        st.code("python task2/run_pruning.py --evals_dir data/Evals --keep_ratio 0.3 --output_dir task2/results", language="bash")
        return

    benchmarks = st.multiselect("Benchmark", sorted(summary["benchmark"].unique()), default=sorted(summary["benchmark"].unique()))
    methods = st.multiselect("Method", sorted(summary["method"].unique()), default=["difficulty_aware"] if "difficulty_aware" in summary["method"].unique() else sorted(summary["method"].unique()))
    sf = summary[summary["benchmark"].isin(benchmarks) & summary["method"].isin(methods)]
    cf = comparison[comparison["benchmark"].isin(benchmarks) & comparison["method"].isin(methods)]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean Spearman", f"{sf['spearman'].mean():.3f}")
    c2.metric("Mean MAE", f"{sf['mae'].mean():.4f}")
    c3.metric("Pairwise agreement", f"{sf['pairwise_agreement'].mean():.3f}")
    c4.metric("Avg cost reduction", f"{sf['cost_reduction_pct'].mean():.1f}%")

    st.subheader("Validation metrics")
    st.dataframe(sf, use_container_width=True)
    st.subheader("Full vs pruned scores")
    st.plotly_chart(px.scatter(cf, x="full_score", y="pruned_score", color="model", symbol="method", facet_col="benchmark", title="Full benchmark score vs pruned benchmark score"), use_container_width=True)
    st.dataframe(cf, use_container_width=True)

    if not selected.empty:
        st.subheader("Selected sample counts")
        counts = selected.groupby(["benchmark","method"], as_index=False)["sample_id"].nunique().rename(columns={"sample_id":"selected_samples"})
        st.dataframe(counts, use_container_width=True)

    for filename in ["full_scores.csv", "pruned_scores.csv", "comparison.csv", "summary_metrics.csv", "selected_samples.csv", "leave_one_model_out.csv"]:
        path = results / filename
        if path.exists():
            st.download_button(f"Download {filename}", path.read_bytes(), filename)

    st.info("Main method: difficulty-aware stratified pruning. Sample difficulty is 1 - average model score, then selected by preserving difficulty and category strata.")


def render_mmmu_probe(project_root: Path) -> None:
    results = project_root / "task2" / "results"
    csv_path = results / "mmmu_probe_results.csv"
    json_path = results / "mmmu_probe_summary.json"
    if not csv_path.exists():
        st.warning("MMMU probe results were not found yet. Run the working probe command below.")
        st.code("python task2/mmmu_probe/run_mmmu_probe.py --evals_dir data/Evals --output_dir task2/results", language="bash")
        return
    df = pd.read_csv(csv_path)
    summary = json.loads(json_path.read_text()) if json_path.exists() else {}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Samples processed", summary.get("samples_processed", len(df)))
    c2.metric("Samples with images", summary.get("samples_with_images", int((df.get('num_images', 0)>0).sum()) if 'num_images' in df else 0))
    c3.metric("Probe accuracy", "n/a" if summary.get("probe_accuracy_when_target_available") is None else f"{summary['probe_accuracy_when_target_available']:.3f}")
    c4.metric("Mean review score", "n/a" if summary.get("mean_review_score") is None else f"{summary['mean_review_score']:.3f}")
    if "category" in df.columns and "probe_acc" in df.columns:
        st.plotly_chart(px.bar(df.groupby("category", as_index=False)["probe_acc"].mean(), x="category", y="probe_acc", title="Probe accuracy by MMMU category"), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download MMMU probe results", csv_path.read_bytes(), "mmmu_probe_results.csv")


def render_docs(project_root: Path) -> None:
    docs = [
        project_root / "README.md",
        project_root / "docs" / "task1_design.md",
        project_root / "task2" / "docs" / "task2_methodology.md",
        project_root / "task2" / "README_Task2.md",
    ]
    choice = st.selectbox("Document", [p.name for p in docs if p.exists()])
    path = next(p for p in docs if p.name == choice)
    st.markdown(path.read_text(encoding="utf-8"))
