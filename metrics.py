from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def normalize_column_name(col: object) -> str:
    name = str(col).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "unnamed"


def infer_model_profile(path_or_name: str) -> dict:
    text = str(path_or_name).replace("\\", "/")
    patterns = [
        r"Model[_\s-]*(?P<model>[A-Za-z0-9]+)[_\s-]*profile[_\s-]*(?P<profile>\d+)",
        r"(?P<model>model[_\s-]*[A-Za-z0-9]+).*?profile[_\s-]*(?P<profile>\d+)",
    ]
    model_token = None
    profile = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            model_token = m.group("model")
            profile = m.group("profile")
            break
    if model_token is None:
        stem = Path(text).stem
        model_token = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "unknown"
    model_token = str(model_token).replace("model_", "").replace("Model_", "")
    if profile is None:
        m = re.search(r"profile[_\s-]*(\d+)", text, flags=re.IGNORECASE)
        profile = m.group(1) if m else "unknown"
    model_id = f"model_{model_token}".lower()
    model_name = f"Model {model_token.upper()}" if len(model_token) == 1 else f"Model {model_token}"
    profile_id = f"profile_{profile}"
    return {"model_id": model_id, "model_name": model_name, "profile_id": profile_id, "profile_number": profile}


def _detect_header(raw: pd.DataFrame) -> int:
    keywords = {"input length", "output length", "throughput", "ttft", "batch size"}
    best_idx = 0
    best_score = -1
    for idx, row in raw.head(15).iterrows():
        values = {str(x).strip().lower() for x in row.dropna().tolist()}
        score = sum(any(k in v for v in values) for k in keywords)
        if score > best_score:
            best_score, best_idx = score, idx
    return int(best_idx)


def parse_perf_excel(file_obj, source_name: Optional[str] = None) -> pd.DataFrame:
    source_name = source_name or getattr(file_obj, "name", "uploaded.xlsx")
    xls = pd.ExcelFile(file_obj)
    frames = []
    meta = infer_model_profile(source_name)
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        if raw.dropna(how="all").empty:
            continue
        header_idx = _detect_header(raw)
        header = raw.iloc[header_idx].fillna("").tolist()
        df = raw.iloc[header_idx + 1 :].copy()
        df.columns = [normalize_column_name(c) for c in header]
        df = df.dropna(how="all")
        if df.empty:
            continue
        # remove repeated header rows if present
        first_col = df.columns[0]
        df = df[df[first_col].astype(str).str.lower() != str(header[0]).lower()]
        for key, value in meta.items():
            df[key] = value
        df["source_filename"] = Path(source_name).name
        df["source_path"] = str(source_name)
        df["sheet_name"] = sheet
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return coerce_numeric_columns(out)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    protected = {"model_id", "model_name", "profile_id", "profile_number", "source_filename", "source_path", "sheet_name"}
    for col in out.columns:
        if col not in protected:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() >= max(1, int(0.4 * len(out))):
                out[col] = converted
    return out


def load_default_perf_data(root: str | Path = "data/perf_data") -> pd.DataFrame:
    root = Path(root)
    files = sorted(root.rglob("*.xlsx")) if root.exists() else []
    frames = []
    for path in files:
        try:
            frames.append(parse_perf_excel(path, str(path)))
        except Exception as exc:
            frames.append(pd.DataFrame({"source_filename": [path.name], "parse_error": [str(exc)]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_uploaded_perf_data(uploaded_files: Optional[Iterable]) -> pd.DataFrame:
    if not uploaded_files:
        return pd.DataFrame()
    frames = []
    for f in uploaded_files:
        try:
            frames.append(parse_perf_excel(f, getattr(f, "name", "uploaded.xlsx")))
        except Exception as exc:
            frames.append(pd.DataFrame({"source_filename": [getattr(f, "name", "uploaded.xlsx")], "parse_error": [str(exc)]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stable_sample_id(row: pd.Series) -> str:
    payload = "|".join(str(row.get(c, "")) for c in sorted(row.index))
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]
