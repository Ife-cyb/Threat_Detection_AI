import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

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


def train(model_name: str = "random_forest", use_mlflow: bool = True) -> None:
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
    
    # Start MLflow tracking if available
    if use_mlflow and HAS_MLFLOW:
        mlflow.set_experiment("threat_detection")
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            if model_name == "random_forest":
                mlflow.log_params({
                    "model_type": "RandomForest",
                    "n_estimators": 400,
                    "max_depth": "None",
                    "min_samples_split": 4,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced_subsample",
                })
            elif model_name == "xgboost":
                mlflow.log_params({
                    "model_type": "XGBoost",
                    "n_estimators": 400,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                })
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='binary')
            recall = recall_score(y_val, y_pred, average='binary')
            f1 = f1_score(y_val, y_pred, average='binary')
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            })
            
            # Log model
            mlflow.sklearn.log_model(model, f"model_{model_name}")
            
            # Log artifact
            report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], digits=4)
            print(report)
            
            report_path = MODELS_DIR / f"val_report_{model_name}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(str(report_path))
    else:
        # Train without MLflow
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], digits=4)
        print(report)
        
        with open(MODELS_DIR / f"val_report_{model_name}.txt", "w", encoding="utf-8") as f:
            f.write(report)

    joblib.dump(model, MODELS_DIR / f"model_{model_name}.joblib")
    print(f"Saved model to {MODELS_DIR / f'model_{model_name}.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline classifier on UNSW-NB15")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()
    train(args.model, use_mlflow=not args.no_mlflow) 