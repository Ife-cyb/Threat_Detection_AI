import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed/unsw_nb15")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, test_df


def get_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    return X, y


def build_model(model_name: str) -> object:
    if model_name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not available. Install xgboost or choose random_forest.")
        return XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        )
    # Default: RandomForest
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def train(model_name: str = "random_forest") -> None:
    artifacts = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor = artifacts["preprocessor"]

    train_df, _ = load_processed()
    X_df, y = get_features_targets(train_df)

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)

    model = build_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], digits=4)
    print(report)

    joblib.dump(model, MODELS_DIR / f"model_{model_name}.joblib")
    with open(MODELS_DIR / f"val_report_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved model to {MODELS_DIR / f'model_{model_name}.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline classifier on UNSW-NB15")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    args = parser.parse_args()
    train(args.model) 