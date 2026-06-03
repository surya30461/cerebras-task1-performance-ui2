from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd


def model_scores(scores: pd.DataFrame, selected: pd.DataFrame | None = None, method: str = "full") -> pd.DataFrame:
    df = scores.copy()
    if selected is not None:
        keys = selected[["benchmark", "sample_id"]].drop_duplicates()
        df = df.merge(keys, on=["benchmark", "sample_id"], how="inner")
    out = df.groupby(["benchmark", "model"], as_index=False)["score"].mean()
    out["method"] = method
    return out


def _spearman(a, b) -> float:
    ar = pd.Series(a).rank().to_numpy(); br = pd.Series(b).rank().to_numpy()
    if len(ar) < 2 or np.std(ar) == 0 or np.std(br) == 0: return float("nan")
    return float(np.corrcoef(ar, br)[0,1])


def _kendall(a, b) -> float:
    n=0; concord=0; discord=0
    for i,j in combinations(range(len(a)),2):
        da=np.sign(a[i]-a[j]); db=np.sign(b[i]-b[j])
        if da==0 or db==0: continue
        n+=1
        if da==db: concord+=1
        else: discord+=1
    return float((concord-discord)/n) if n else float("nan")


def pairwise_agreement(full, pruned) -> float:
    agree=0; total=0
    models=list(full.index)
    for m1,m2 in combinations(models,2):
        if m1 not in pruned.index or m2 not in pruned.index: continue
        total+=1
        agree += int(np.sign(full[m1]-full[m2]) == np.sign(pruned[m1]-pruned[m2]))
    return agree/total if total else float("nan")


def compare_scores(full_scores: pd.DataFrame, pruned_scores: pd.DataFrame, selected: pd.DataFrame, total_samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_clean = full_scores.drop(columns=["method"], errors="ignore").rename(columns={"score":"full_score"})
    pruned_clean = pruned_scores.rename(columns={"score":"pruned_score"})[["benchmark","model","pruned_score","method"]]
    merged = full_clean.merge(pruned_clean, on=["benchmark","model"], how="inner")
    merged["abs_error"]=(merged["full_score"]-merged["pruned_score"]).abs()
    rows=[]
    for (bench, method), g in merged.groupby(["benchmark","method"]):
        f=g.set_index("model")["full_score"]; p=g.set_index("model")["pruned_score"]
        kept=selected[selected["benchmark"]==bench]["sample_id"].nunique()
        total=total_samples[total_samples["benchmark"]==bench]["sample_id"].nunique()
        rows.append({
            "benchmark": bench, "method": method,
            "spearman": _spearman(f.values, p.values),
            "kendall_tau": _kendall(f.values, p.values),
            "mae": g["abs_error"].mean(),
            "max_abs_error": g["abs_error"].max(),
            "top1_preserved": f.idxmax()==p.idxmax(),
            "pairwise_agreement": pairwise_agreement(f, p),
            "samples_kept": kept, "total_samples": total,
            "cost_reduction_pct": 100*(1-kept/max(total,1)),
        })
    return merged, pd.DataFrame(rows)
