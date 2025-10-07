"""
Retraining pipeline for automated model updates.
Supports incremental data loading and model comparison.
"""
import argparse
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

from train import build_model, get_features_targets

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed/unsw_nb15")


def load_new_data(new_data_path: str) -> pd.DataFrame:
    """Load new data for retraining."""
    return pd.read_csv(new_data_path)


def combine_datasets(original_df: pd.DataFrame, new_df: pd.DataFrame, max_samples: int = None) -> pd.DataFrame:
    """Combine original and new datasets, optionally limiting total samples."""
    combined = pd.concat([original_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates()
    if max_samples and len(combined) > max_samples:
        # Keep most recent samples
        combined = combined.tail(max_samples)
    return combined


def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average='binary'),
        "recall": recall_score(y, y_pred, average='binary'),
        "f1_score": f1_score(y, y_pred, average='binary'),
    }


def retrain(
    model_name: str = "random_forest",
    new_data_path: str = None,
    auto_promote: bool = False,
    use_mlflow: bool = True,
    max_samples: int = None,
) -> None:
    """
    Retrain model with optional new data.
    
    Args:
        model_name: Model type to retrain
        new_data_path: Path to new labeled data (optional)
        auto_promote: Automatically promote if new model is better
        use_mlflow: Use MLflow for tracking
        max_samples: Maximum samples to keep in training data
    """
    # Load existing artifacts
    artifacts = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor = artifacts["preprocessor"]
    
    # Load existing model for comparison
    old_model_path = MODELS_DIR / f"model_{model_name}.joblib"
    old_model = None
    if old_model_path.exists():
        old_model = joblib.load(old_model_path)
        print(f"Loaded existing model from {old_model_path}")
    
    # Load training data
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    
    # Combine with new data if provided
    if new_data_path:
        new_df = load_new_data(new_data_path)
        train_df = combine_datasets(train_df, new_df, max_samples=max_samples)
        print(f"Combined training data: {len(train_df)} samples")
    
    # Prepare data
    X_df, y = get_features_targets(train_df)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    
    # Train new model
    new_model = build_model(model_name)
    new_model.fit(X_train, y_train)
    
    # Evaluate both models
    new_metrics = evaluate_model(new_model, X_val, y_val)
    print(f"\nNew model metrics:")
    for k, v in new_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    old_metrics = None
    if old_model:
        old_metrics = evaluate_model(old_model, X_val, y_val)
        print(f"\nOld model metrics:")
        for k, v in old_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Track with MLflow
    if use_mlflow and HAS_MLFLOW:
        mlflow.set_experiment("threat_detection_retrain")
        with mlflow.start_run(run_name=f"{model_name}_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "model_name": model_name,
                "new_data": new_data_path if new_data_path else "none",
                "total_samples": len(train_df),
            })
            mlflow.log_metrics({
                f"new_{k}": v for k, v in new_metrics.items()
            })
            if old_metrics:
                mlflow.log_metrics({
                    f"old_{k}": v for k, v in old_metrics.items()
                })
            mlflow.sklearn.log_model(new_model, f"retrained_{model_name}")
    
    # Decide whether to promote
    promote = auto_promote
    if old_metrics and not auto_promote:
        # Compare F1 scores
        if new_metrics["f1_score"] > old_metrics["f1_score"]:
            print(f"\nNew model F1 ({new_metrics['f1_score']:.4f}) > Old model F1 ({old_metrics['f1_score']:.4f})")
            response = input("Promote new model? (y/n): ")
            promote = response.lower() == 'y'
    
    if promote or not old_model:
        # Save new model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if old_model_path.exists():
            backup_path = MODELS_DIR / f"model_{model_name}_backup_{timestamp}.joblib"
            old_model_path.rename(backup_path)
            print(f"Backed up old model to {backup_path}")
        
        joblib.dump(new_model, old_model_path)
        print(f"Promoted new model to {old_model_path}")
        
        # Update training data if new data was added
        if new_data_path:
            train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
            print(f"Updated training data at {PROCESSED_DIR / 'train.csv'}")
    else:
        print("New model not promoted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain threat detection model")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument("--new-data", type=str, help="Path to new labeled data CSV")
    parser.add_argument("--auto-promote", action="store_true", help="Automatically promote new model")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to keep in training data")
    args = parser.parse_args()
    
    retrain(
        model_name=args.model,
        new_data_path=args.new_data,
        auto_promote=args.auto_promote,
        use_mlflow=not args.no_mlflow,
        max_samples=args.max_samples,
    )
