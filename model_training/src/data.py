from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import CFG

# ---------------------------------------------------------------------------
# Label / clinical feature mappings
# ---------------------------------------------------------------------------
LABEL_MAP = {"Cardiomegaly": 1, "No Finding": 0}
SEX_MAP   = {"F": 1.0, "M": 0.0}   # Female=1, Male=0, unknown → 0.5 (neutral)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels(csv_path: str, image_dir: str) -> pd.DataFrame:
    """Read the CSV, map labels to binary, attach clinical features, resolve paths.

    Expected CSV columns : 'Image Index', 'Finding Labels'
    Optional CSV columns : 'Patient Age', 'Patient Sex'

    Returned DataFrame columns:
        image_name, label_raw, age_norm, sex_bin, label, image_path
    """
    df = pd.read_csv(csv_path)

    required_cols = ["Image Index", "Finding Labels"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    has_clinical = {"Patient Age", "Patient Sex"}.issubset(df.columns)
    keep_cols = ["Image Index", "Finding Labels"] + (
        ["Patient Age", "Patient Sex"] if has_clinical else []
    )
    df = df[keep_cols].copy()

    if has_clinical:
        df.columns = ["image_name", "label_raw", "age_raw", "sex_raw"]
        df["age_norm"] = (
            pd.to_numeric(df["age_raw"], errors="coerce").clip(0, 100) / 100.0
        ).fillna(0.5).astype(np.float32)
        df["sex_bin"] = df["sex_raw"].map(SEX_MAP).fillna(0.5).astype(np.float32)
    else:
        df.columns = ["image_name", "label_raw"]
        df["age_norm"] = np.float32(0.5)
        df["sex_bin"]  = np.float32(0.5)
        print("Warning: Patient Age / Sex not found — using neutral defaults (0.5).")

    df["image_name"] = df["image_name"].astype(str).str.strip()

    df["label"] = df["label_raw"].map(LABEL_MAP)
    unknown = df["label"].isna().sum()
    if unknown:
        unseen = df.loc[df["label"].isna(), "label_raw"].unique().tolist()
        print(f"Warning: {unknown} rows with unknown labels {unseen} — skipped.")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    df["image_path"] = df["image_name"].apply(lambda x: os.path.join(image_dir, x))

    missing_imgs = (~df["image_path"].apply(os.path.exists)).sum()
    if missing_imgs:
        print(f"Warning: {missing_imgs} images not found on disk — skipped.")

    df = df[df["image_path"].apply(os.path.exists)].copy()
    df = df.drop_duplicates(subset=["image_name"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid labelled images found.")

    print(f"Loaded {len(df)} rows")
    print(df["label"].value_counts().sort_index().rename({0: "No Finding", 1: "Cardiomegaly"}))
    if has_clinical:
        print(
            f"Age range: {df['age_raw'].min():.0f}–{df['age_raw'].max():.0f}  |  "
            f"Sex: {df['sex_raw'].value_counts().to_dict()}"
        )
    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

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

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def estimate_gray_mean_std(
    df: pd.DataFrame,
    sample_size: Optional[int] = 1000,
    img_size: int | None = None,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Compute grayscale mean and std from a random sample of training images."""
    img_size = img_size if img_size is not None else CFG.img_size
    seed     = seed     if seed     is not None else CFG.seed

    sample_df = df
    if sample_size is not None and len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=seed)

    pixel_sum, pixel_sq_sum, pixel_count = 0.0, 0.0, 0
    for path in sample_df["image_path"]:
        arr = np.asarray(
            Image.open(path).convert("L").resize((img_size, img_size)),
            dtype=np.float32,
        ) / 255.0
        pixel_sum    += arr.sum()
        pixel_sq_sum += (arr ** 2).sum()
        pixel_count  += arr.size

    mean = pixel_sum / pixel_count
    std  = float(np.sqrt(max(pixel_sq_sum / pixel_count - mean ** 2, 1e-12)))
    return float(mean), std
