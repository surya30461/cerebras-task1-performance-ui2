from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_benchmark_model(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        bench, model = stem.split("__", 1)
    else:
        parts = path.parts
        model = parts[-2] if len(parts) >= 2 else "unknown_model"
        bench = stem
    bench = bench.replace("_v5", "") if bench == "live_code_bench_v5" else bench
    return bench, model


def extract_score(obj: dict[str, Any]) -> tuple[float | None, str | None]:
    ss = obj.get("sample_score", {}) or {}
    score = ss.get("score", ss)
    name = None
    value = None
    if isinstance(score, dict):
        name = score.get("main_score_name") or ss.get("main_score_name")
        val = score.get("value", score)
        if isinstance(val, dict):
            if name and name in val:
                value = val.get(name)
            elif "pass" in val:
                name, value = "pass", val.get("pass")
            elif "acc" in val:
                name, value = "acc", val.get("acc")
            else:
                for k, v in val.items():
                    if isinstance(v, (int, float, bool)):
                        name, value = k, v
                        break
        elif isinstance(val, (int, float, bool)):
            name, value = name or "score", val
    elif isinstance(score, (int, float, bool)):
        name, value = "score", score
    if isinstance(value, bool):
        value = 1.0 if value else 0.0
    try:
        return float(value), name
    except Exception:
        return None, name


def extract_metadata(obj: dict[str, Any]) -> dict[str, Any]:
    ss = obj.get("sample_score", {}) or {}
    smd = ss.get("sample_metadata", {}) if isinstance(ss, dict) else {}
    md = obj.get("metadata", {}) or {}
    question = smd.get("question") or md.get("question") or obj.get("input") or ""
    category = smd.get("category") or md.get("category") or md.get("subject") or md.get("subdomain")
    return {"question": str(question)[:800], "category": category or "default"}


def load_review_scores(evals_dir: str | Path) -> pd.DataFrame:
    root = Path(evals_dir)
    files = list(root.rglob("reviews/**/*.jsonl")) + list(root.rglob("Part 1/reviews/*.jsonl"))
    seen = set(); rows = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        benchmark, model = parse_benchmark_model(path)
        # normalize MMMU files into one benchmark while preserving category
        if benchmark.startswith("mmmu_"):
            category_from_file = benchmark.replace("mmmu_", "")
            benchmark = "mmmu"
        else:
            category_from_file = None
        for obj in read_jsonl(path):
            score, score_name = extract_score(obj)
            if score is None:
                continue
            meta = extract_metadata(obj)
            sample_id = str(obj.get("index", obj.get("sample_id", len(rows))))
            category = category_from_file or meta.get("category", "default")
            rows.append({
                "benchmark": benchmark,
                "model": obj.get("model", model),
                "sample_id": sample_id,
                "score": score,
                "score_name": score_name or "score",
                "category": category,
                "question": meta.get("question", ""),
                "source_file": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            })
    return pd.DataFrame(rows)


def load_prediction_index(evals_dir: str | Path) -> pd.DataFrame:
    root = Path(evals_dir)
    files = list(root.rglob("predictions/**/*.jsonl")) + list(root.rglob("Part 1/predictions/*.jsonl"))
    rows=[]; seen=set()
    for path in files:
        if path in seen: continue
        seen.add(path)
        benchmark, model = parse_benchmark_model(path)
        if benchmark.startswith("mmmu_"):
            category = benchmark.replace("mmmu_", "")
            benchmark = "mmmu"
        else:
            category = "default"
        for obj in read_jsonl(path):
            rows.append({"benchmark": benchmark, "model": obj.get("model", model), "sample_id": str(obj.get("index", len(rows))), "category": category, "source_file": str(path)})
    return pd.DataFrame(rows)
