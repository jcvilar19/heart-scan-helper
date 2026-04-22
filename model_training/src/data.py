from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CFG


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------
FILENAME_CANDIDATES = [
    "image_name", "filename", "file", "image", "image_id", "img", "name",
    "image index", "image_index",                    # NIH ChestX-ray14
]
LABEL_CANDIDATES = [
    "label", "cardiomegaly", "class", "target", "y",
    "finding_labels", "finding labels", "finding",   # NIH ChestX-ray14
    "labels",
]
POSITIVE_KEYWORD = "cardiomegaly"


def _autodetect(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column in *df* whose lowercase name is in *candidates*."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")


def _coerce_to_binary(series: pd.Series) -> pd.Series:
    """Map mixed label encodings (0/1, 'cardiomegaly', 'no finding', bool, ...) to 0/1."""
    def to_int(v):
        if pd.isna(v):
            return 0
        if isinstance(v, (int, np.integer)):
            return int(v != 0)
        if isinstance(v, (float, np.floating)):
            return int(v != 0)
        if isinstance(v, bool):
            return int(v)
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "positive", "pos"}:
            return 1
        if s in {"0", "false", "no", "n", "negative", "neg", "no finding", ""}:
            return 0
        return int(POSITIVE_KEYWORD in s)
    return series.apply(to_int).astype(int)


def _resolve_filenames(df: pd.DataFrame, filename_col: str, image_dir: str) -> pd.DataFrame:
    """Add an `image_path` column. Drops rows whose file cannot be found.

    Tolerates different case, trailing spaces, and missing/wrong extensions.
    """
    disk: dict[str, str] = {}
    for entry in os.scandir(image_dir):
        if not entry.is_file():
            continue
        name = entry.name
        disk[name.lower()] = name
        stem = os.path.splitext(name)[0].lower()
        disk.setdefault(stem, name)

    resolved, missing = [], []
    for fn in df[filename_col].astype(str):
        raw = fn.strip()
        raw_l = raw.lower()
        hit = disk.get(raw_l) or disk.get(os.path.splitext(raw_l)[0])
        if hit is None:
            for ext in (".png", ".jpg", ".jpeg"):
                if raw_l + ext in disk:
                    hit = disk[raw_l + ext]
                    break
        if hit is None:
            missing.append(raw)
            resolved.append(None)
        else:
            resolved.append(os.path.join(image_dir, hit))

    df = df.copy()
    df["image_path"] = resolved
    keep = df["image_path"].notna()
    if (~keep).any():
        print(f"Warning: {(~keep).sum()} rows dropped (file not found). Examples: {missing[:5]}")
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_labels(csv_path: str, image_dir: str) -> pd.DataFrame:
    """Read CSV, auto-detect filename + label columns, coerce labels, resolve paths.

    Returned DataFrame columns: filename, label, image_path
    """
    df = pd.read_csv(csv_path)
    fn_col = _autodetect(df, FILENAME_CANDIDATES)
    lb_col = _autodetect(df, LABEL_CANDIDATES)
    print(f"Detected filename column: {fn_col!r}   label column: {lb_col!r}")

    df = df[[fn_col, lb_col]].rename(columns={fn_col: "filename", lb_col: "label"})
    df["label"] = _coerce_to_binary(df["label"])
    df = _resolve_filenames(df, "filename", image_dir)
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid labelled images found.")

    n_pos = int(df["label"].sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"Loaded {len(df)} labelled images   pos={n_pos}   neg={n_neg}")
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_size: float | None = None,
    test_size: float | None = None,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split.

    Falls back to CFG values when parameters are not supplied.
    """
    val_size  = val_size  if val_size  is not None else CFG.val_size
    test_size = test_size if test_size is not None else CFG.test_size
    seed      = seed      if seed      is not None else CFG.seed

    train_tmp_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed,
    )
    rel_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_tmp_df, test_size=rel_val,
        stratify=train_tmp_df["label"], random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
