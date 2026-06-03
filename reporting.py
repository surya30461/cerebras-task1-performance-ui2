from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def save_selected_jsons(selected: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for bench, g in selected.groupby("benchmark"):
        ids = sorted(g["sample_id"].astype(str).unique().tolist())
        aliases = {"live_code_bench": "live_code_bench", "aa_lcr": "aa_lcr", "mmmu": "mmmu"}
        name = aliases.get(bench, bench)
        with (out / f"{name}_selected_samples.json").open("w", encoding="utf-8") as f:
            json.dump({"benchmark": bench, "sample_ids": ids}, f, indent=2)


def save_plots(full_scores: pd.DataFrame, comparison: pd.DataFrame, selected: pd.DataFrame, scores_with_diff: pd.DataFrame, output_dir: str | Path) -> None:
    plot_dir = Path(output_dir) / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        pivot = comparison.pivot_table(index="model", columns="method", values="pruned_score", aggfunc="mean")
        pivot["full"] = comparison.groupby("model")["full_score"].mean()
        ax = pivot.plot(kind="bar", figsize=(10,5), title="Full vs Pruned Scores")
        ax.set_ylabel("Score")
        plt.tight_layout(); plt.savefig(plot_dir / "full_vs_pruned_scores.png"); plt.close()
    except Exception: pass
    try:
        c = comparison.copy()
        c["full_rank"] = c.groupby(["benchmark","method"])["full_score"].rank(ascending=False)
        c["pruned_rank"] = c.groupby(["benchmark","method"])["pruned_score"].rank(ascending=False)
        ax = c.plot.scatter(x="full_rank", y="pruned_rank", figsize=(6,5), title="Rank Comparison")
        plt.tight_layout(); plt.savefig(plot_dir / "rank_comparison.png"); plt.close()
    except Exception: pass
    try:
        ax = comparison.groupby("model")["abs_error"].mean().sort_values().plot(kind="bar", figsize=(8,4), title="Mean Absolute Error by Model")
        ax.set_ylabel("MAE")
        plt.tight_layout(); plt.savefig(plot_dir / "error_by_model.png"); plt.close()
    except Exception: pass
    try:
        if "difficulty" in scores_with_diff.columns:
            samples = scores_with_diff.drop_duplicates(["benchmark","sample_id"])
            ax = samples["difficulty"].hist(figsize=(7,4))
            ax.set_title("Sample Difficulty Distribution"); ax.set_xlabel("Difficulty = 1 - average score")
            plt.tight_layout(); plt.savefig(plot_dir / "sample_difficulty_distribution.png"); plt.close()
    except Exception: pass
