from __future__ import annotations
import argparse

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] if _Path(__file__).parent.name != "mmmu_probe" else _Path(__file__).resolve().parents[2]))
from pathlib import Path
import pandas as pd

from task2.load_evals import load_review_scores
from task2.pruning.difficulty_estimator import add_sample_difficulty
from task2.pruning.sample_selector import select_samples
from task2.evaluate_subset import model_scores, compare_scores
from task2.reporting import save_selected_jsons, save_plots


def run(evals_dir: str, keep_ratio: float, output_dir: str, seed: int = 42) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    scores = load_review_scores(evals_dir)
    if scores.empty:
        raise SystemExit(f"No review scores found under {evals_dir}")
    scores = add_sample_difficulty(scores)
    full = model_scores(scores, method="full")
    methods = ["random", "uniform_stratified", "difficulty_aware"]
    all_selected=[]; all_pruned=[]; all_comp=[]; all_summary=[]
    total_samples = scores.drop_duplicates(["benchmark","sample_id"])
    for method in methods:
        selected = select_samples(scores, method=method, keep_ratio=keep_ratio, seed=seed)
        selected["method"] = method
        pruned = model_scores(scores, selected, method=method)
        comparison, summary = compare_scores(full, pruned, selected, total_samples)
        all_selected.append(selected); all_pruned.append(pruned); all_comp.append(comparison); all_summary.append(summary)
    selected_all = pd.concat(all_selected, ignore_index=True)
    pruned_all = pd.concat(all_pruned, ignore_index=True)
    comparison_all = pd.concat(all_comp, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)

    main_selected = selected_all[selected_all["method"] == "difficulty_aware"].drop(columns=["method"], errors="ignore")
    save_selected_jsons(main_selected, out)
    full.to_csv(out / "full_scores.csv", index=False)
    pruned_all.to_csv(out / "pruned_scores.csv", index=False)
    comparison_all.to_csv(out / "comparison.csv", index=False)
    summary_all.to_csv(out / "summary_metrics.csv", index=False)
    selected_all.to_csv(out / "selected_samples.csv", index=False)
    scores.to_csv(out / "sample_scores_with_difficulty.csv", index=False)

    # Leave-one-model-out validation for main method.
    rows=[]
    for model in scores["model"].unique():
        train = add_sample_difficulty(scores, holdout_model=model)
        sel = select_samples(train[train["model"] != model], method="difficulty_aware", keep_ratio=keep_ratio, seed=seed)
        held = scores[scores["model"] == model]
        full_h = held.groupby("benchmark")["score"].mean()
        pruned_h = held.merge(sel[["benchmark","sample_id"]], on=["benchmark","sample_id"], how="inner").groupby("benchmark")["score"].mean()
        for bench in full_h.index:
            if bench in pruned_h.index:
                rows.append({"held_out_model": model, "benchmark": bench, "full_score": full_h[bench], "pruned_score": pruned_h[bench], "abs_error": abs(full_h[bench]-pruned_h[bench])})
    pd.DataFrame(rows).to_csv(out / "leave_one_model_out.csv", index=False)
    save_plots(full, comparison_all, main_selected, scores, out)
    print(f"Wrote Task 2 results to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evals_dir", default="data/Evals")
    p.add_argument("--keep_ratio", type=float, default=0.3)
    p.add_argument("--output_dir", default="task2/results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.evals_dir, args.keep_ratio, args.output_dir, args.seed)
