import argparse
import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = Path("data/raw/unsw_nb15")
PROCESSED_DIR = Path("data/processed/unsw_nb15")
MODELS_DIR = Path("models")

UNSW_GITHUB_MIRROR = {
    "UNSW-NB15_1.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/UNSW-NB15_1.csv.zip",
    "UNSW-NB15_2.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/UNSW-NB15_2.csv.zip",
    "UNSW-NB15_3.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/UNSW-NB15_3.csv.zip",
    "UNSW-NB15_4.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/UNSW-NB15_4.csv.zip",
    "UNSW_NB15_training-set.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/sets/UNSW_NB15_training-set.csv.zip",
    "UNSW_NB15_testing-set.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/sets/UNSW_NB15_testing-set.csv.zip",
    "NUSW-NB15_groundTruth.csv.zip": "https://raw.githubusercontent.com/iammyr/encrypted-network-datasets/master/NUSW-NB15_groundTruth.csv.zip",
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
    if dest_path.suffix == ".zip" and dest_path.stat().st_size < 1024:
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file {dest_path} appears invalid (too small)")


def maybe_extract_zip(zip_path: Path, extract_dir: Path) -> Optional[List[Path]]:
    if zip_path.suffix != ".zip":
        return None
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".csv"):
                zf.extract(member, extract_dir)
                extracted.append(extract_dir / member)
    return extracted


def kaggle_download(dataset: str, files: List[str], dest_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError("kaggle package not installed. Install requirements and set credentials.") from exc
    api = KaggleApi()
    api.authenticate()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        api.dataset_download_file(dataset=dataset, file_name=fname, path=str(dest_dir), force=False, quiet=False)
        zip_path = dest_dir / f"{fname}.zip"
        if zip_path.exists():
            maybe_extract_zip(zip_path, dest_dir)
        else:
            if not (dest_dir / fname).exists():
                raise FileNotFoundError(f"Expected {fname} or {fname}.zip from Kaggle, not found in {dest_dir}")


def find_or_download_unsw(files: List[str]) -> List[Path]:
    local_paths: List[Path] = []
    for fname in files:
        zip_path = RAW_DIR / fname
        csv_candidate = RAW_DIR / fname.replace(".zip", "")
        if csv_candidate.exists():
            local_paths.append(csv_candidate)
            continue
        if not zip_path.exists():
            url = UNSW_GITHUB_MIRROR.get(fname)
            if url is None:
                raise FileNotFoundError(f"No URL known for {fname}")
            try:
                download_file(url, zip_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to download {fname} from {url}: {exc}")
        extracted = maybe_extract_zip(zip_path, RAW_DIR)
        if extracted:
            for p in extracted:
                if p.suffix == ".csv":
                    target = RAW_DIR / Path(p.name)
                    if p != target:
                        os.replace(p, target)
                    local_paths.append(target)
                    break
        elif csv_candidate.exists():
            local_paths.append(csv_candidate)
        else:
            raise RuntimeError(f"Could not extract CSV from {zip_path}")
    return local_paths


def load_unsw(use_presplit: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    ensure_dirs()
    if use_presplit:
        train_path = None
        test_path = None
        # Try training file first
        try:
            train_path = find_or_download_unsw(["UNSW_NB15_training-set.csv.zip"])[0]
        except Exception:
            train_path = None
        # Try testing file
        try:
            test_path = find_or_download_unsw(["UNSW_NB15_testing-set.csv.zip"])[0]
        except Exception:
            test_path = None
        if train_path is not None and test_path is not None:
            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)
            return train_df, test_df
        if train_path is not None and test_path is None:
            # Proceed with only training; caller will split test
            train_df = pd.read_csv(train_path, low_memory=False)
            return train_df, None
        # else fallback to four-part files
    parts = find_or_download_unsw([
        "UNSW-NB15_1.csv.zip",
        "UNSW-NB15_2.csv.zip",
        "UNSW-NB15_3.csv.zip",
        "UNSW-NB15_4.csv.zip",
    ])
    df = pd.concat((pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True)
    if not any(c.lower() == "label" for c in df.columns) and not any(c.lower() == "attack_cat" for c in df.columns):
        try:
            gt_path = find_or_download_unsw(["NUSW-NB15_groundTruth.csv.zip"])[0]
            gt_df = pd.read_csv(gt_path, low_memory=False)
            df = attach_ground_truth(df, gt_df)
        except Exception as exc:
            raise RuntimeError(
                "UNSW-NB15 data without labels detected and ground truth merge failed. "
                "Place labeled splits in data/raw/unsw_nb15/ or ensure network access."
            ) from exc
    return df, None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df


def attach_ground_truth(df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    base = standardize_columns(df)
    gt = standardize_columns(gt_df)
    if "id" not in base.columns or "id" not in gt.columns:
        raise KeyError("'id' column not found to merge ground truth")
    cols = [c for c in gt.columns if c in {"id", "label", "attack_cat"}]
    gt = gt[cols]
    merged = base.merge(gt, on="id", how="left")
    if "label" not in merged.columns and "attack_cat" not in merged.columns:
        raise KeyError("Ground truth did not provide 'label' or 'attack_cat'")
    return merged


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()
    return df


def get_feature_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = standardize_columns(df)
    label_col = "label"
    if label_col not in df.columns:
        if "attack_cat" in df.columns:
            df[label_col] = (df["attack_cat"].fillna("Normal").astype(str) != "Normal").astype(int)
        else:
            raise KeyError("Could not find 'label' or 'attack_cat' to derive target")
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    for col in ["id", "event", "attack_cat"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_candidates = [
        "proto", "service", "state", "source_ip", "dest_ip", "srcip", "dstip"
    ]
    categorical_features = [c for c in categorical_candidates if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=None,
    )
    return preprocessor, numeric_features, categorical_features


def preprocess_and_save(use_presplit: bool = True, test_size: float = 0.2, random_state: int = 42) -> None:
    train_df, test_df = load_unsw(use_presplit=use_presplit)

    train_df = standardize_columns(train_df)
    train_df = clean_dataframe(train_df)

    if test_df is not None:
        test_df = standardize_columns(test_df)
        test_df = clean_dataframe(test_df)
    else:
        full_X, full_y = get_feature_targets(train_df)
        X_train, X_test, y_train, y_test = train_test_split(
            full_X, full_y, test_size=test_size, random_state=random_state, stratify=full_y
        )
        train_df = X_train.copy()
        train_df["label"] = y_train.values
        test_df = X_test.copy()
        test_df["label"] = y_test.values

    X_train, y_train = get_feature_targets(train_df)
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    preprocessor.fit(X_train)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        MODELS_DIR / "preprocessor.joblib",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved preprocessor to {MODELS_DIR / 'preprocessor.joblib'}")
    print(f"Saved processed train to {train_path}")
    print(f"Saved processed test to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess UNSW-NB15 dataset")
    parser.add_argument("--no-presplit", action="store_true", help="Ignore official train/test and perform a new split")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction if re-splitting")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split")
    parser.add_argument("--use-kaggle", action="store_true", help="Download required files via Kaggle before preprocessing")
    parser.add_argument("--kaggle-dataset", type=str, default="galaxy-dl/unsw-nb15", help="Kaggle dataset slug, e.g., 'galaxy-dl/unsw-nb15'")
    parser.add_argument(
        "--kaggle-files",
        nargs="*",
        default=[
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv",
        ],
        help="Specific files to download from the Kaggle dataset",
    )
    args = parser.parse_args()

    if args.use_kaggle:
        kaggle_download(args.kaggle_dataset, args.kaggle_files, RAW_DIR)

    preprocess_and_save(use_presplit=not args.no_presplit, test_size=args.test_size, random_state=args.random_state) 