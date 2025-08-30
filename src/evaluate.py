import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

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


def evaluate(model_name: str) -> None:
    artifacts = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor = artifacts["preprocessor"]

    _, test_df = load_processed()
    X_test_df, y_test = get_features_targets(test_df)
    X_test = preprocessor.transform(X_test_df)

    model = joblib.load(MODELS_DIR / f"model_{model_name}.joblib")

    y_pred = model.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], digits=4)
    print(report)

    with open(MODELS_DIR / f"test_report_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax)
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig_path = MODELS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    args = parser.parse_args()
    evaluate(args.model) 