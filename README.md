# Threat Detection AI (UNSW-NB15)

End-to-end intrusion detection using the UNSW-NB15 dataset: automated download, preprocessing, model training, evaluation, Streamlit dashboard, and FastAPI for integration.

## Features
- Automated dataset download (UNSW-NB15 via Kaggle or a public mirror), preprocessing, train/test split
- **Multi-dataset support**: UNSW-NB15, CICIDS2017, NSL-KDD with unified preprocessing
- Robust preprocessing: categorical encoding (proto, service, state, etc.), numeric scaling
- Baseline models: RandomForest (default) and XGBoost
- **MLflow integration**: Experiment tracking, model versioning, and metrics logging
- **Automated retraining pipeline**: Incremental learning and model comparison
- Metrics: precision, recall, F1-score, confusion matrix (saved as image)
- **Enhanced Streamlit dashboard**: 
  - Multiple analytics tabs with advanced visualizations
  - Feature importance charts
  - ROC curves and probability distributions
  - Model comparison and performance metrics
  - Batch prediction support
- **Optimized FastAPI**: 
  - Standard `/predict` endpoint for single/small batch predictions
  - `/batch_predict` endpoint for efficient large-scale inference
- Optional FastAPI `/predict` endpoint for programmatic inference

## Project Structure
```
threat-detection-ai/
├─ data/
│  ├─ raw/unsw_nb15/              # downloaded CSVs
│  ├─ raw/cicids2017/             # CICIDS2017 dataset (optional)
│  ├─ raw/nsl_kdd/                # NSL-KDD dataset (optional)
│  └─ processed/                  # processed train/test CSVs
├─ models/                        # preprocessor + trained models
├─ mlruns/                        # MLflow tracking data
├─ src/
│  ├─ data_prep.py                # download + preprocess UNSW-NB15
│  ├─ multi_dataset.py            # multi-dataset support (CICIDS2017, NSL-KDD)
│  ├─ train.py                    # model training with MLflow
│  ├─ retrain.py                  # automated retraining pipeline
│  ├─ evaluate.py                 # evaluation + confusion matrix
│  ├─ streamlit_app.py            # Enhanced Streamlit dashboard
│  └─ serve_api.py                # FastAPI service with batch prediction
├─ requirements.txt
└─ README.md
```

## Quickstart
1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Dataset download & preprocessing
- Option A: Kaggle (recommended)
  - Set credentials either via env vars or kaggle.json:
    - Env: `export KAGGLE_USERNAME=your_username; export KAGGLE_KEY=your_key`
    - File: place a `kaggle.json` in `~/.kaggle/kaggle.json` with `{"username":"...","key":"..."}`
  - Download pre-split CSVs and preprocess:
```bash
python src/data_prep.py --use-kaggle --kaggle-dataset galaxy-dl/unsw-nb15 \
  --kaggle-files UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv
```
- Option B: Public mirror fallback
```bash
python src/data_prep.py
# or force re-split if pre-split files are not mirrored
python src/data_prep.py --no-presplit --test-size 0.2
```
Artifacts:
- `models/preprocessor.joblib`
- `data/processed/unsw_nb15/train.csv`, `data/processed/unsw_nb15/test.csv`

3. Train a baseline model:
```bash
python src/train.py --model random_forest
# or
python src/train.py --model xgboost
# Disable MLflow tracking (if needed)
python src/train.py --model random_forest --no-mlflow
```
Artifacts:
- `models/model_<name>.joblib`
- `models/val_report_<name>.txt`
- MLflow experiment tracking in `mlruns/`

4. Evaluate on the test set:
```bash
python src/evaluate.py --model random_forest
```
Artifacts:
- `models/test_report_<name>.txt`
- `models/confusion_matrix_<name>.png`

5. Launch the Streamlit dashboard:
```bash
streamlit run src/streamlit_app.py
```
Features:
- Upload a CSV of network flows (columns like `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, ...)
- View predictions, threat probabilities, attack distribution, and packet/byte stats
- **NEW**: Advanced analytics with feature importance, ROC curves, and probability distributions
- **NEW**: Model performance metrics and comparison
- **NEW**: Batch prediction support for large datasets
- **NEW**: MLflow experiment tracking integration

6. Run the FastAPI service (optional):
```bash
uvicorn src.serve_api:app --reload --host 0.0.0.0 --port 8000
```
Example request (single/small batch):
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "random_forest",
    "records": [
      {"proto":"tcp","service":"http","state":"CON","spkts":10,"dpkts":8,"sbytes":1024,"dbytes":2048},
      {"proto":"udp","service":"dns","state":"INT","spkts":5,"dpkts":5,"sbytes":512,"dbytes":512}
    ]
  }'
```

**NEW**: Batch prediction endpoint for large datasets:
```bash
curl -X POST "http://localhost:8000/batch_predict?model_name=random_forest" \
  -H 'accept: application/json' \
  -F 'file=@/path/to/large_dataset.csv'
```

7. **NEW**: Retrain models with new data:
```bash
# Retrain with new labeled data
python src/retrain.py --model random_forest --new-data path/to/new_data.csv

# Auto-promote if better performance
python src/retrain.py --model random_forest --new-data path/to/new_data.csv --auto-promote

# Limit training set size
python src/retrain.py --model random_forest --max-samples 100000
```

8. **NEW**: Use additional datasets (CICIDS2017, NSL-KDD):
```bash
# Prepare CICIDS2017 dataset
# First, download CICIDS2017 CSV files to data/raw/cicids2017/
python src/multi_dataset.py --dataset cicids2017

# Prepare NSL-KDD dataset
# First, download NSL-KDD files (KDDTrain+.txt, KDDTest+.txt) to data/raw/nsl_kdd/
python src/multi_dataset.py --dataset nsl_kdd

# Then train with the new dataset
python src/train.py --model random_forest
```

9. **NEW**: View MLflow experiments:
```bash
mlflow ui
# Visit http://localhost:5000 to view experiment tracking, model versions, and metrics
```

## Notes on Datasets
- Default is UNSW-NB15 flow-based dataset. Kaggle download is the most reliable.
- If you have only the four-part files `UNSW-NB15_1.csv`..`UNSW-NB15_4.csv`, also include `NUSW-NB15_groundTruth.csv`; the script will merge labels by `id`.
- Target label is `label` (0: Normal, 1: Attack). If missing, it is derived from `attack_cat` (`Normal` vs others).

## Development Tips
- Preprocessor is saved separately to ensure consistent feature handling in apps and API.
- To extend: add feature selection, class-imbalance techniques (e.g., SMOTE), or advanced models.
- **NEW**: MLflow tracks all experiments - use `mlflow ui` to compare model performance
- **NEW**: Retraining pipeline supports incremental learning with automatic model promotion
- **NEW**: Multi-dataset support allows training on different intrusion detection datasets
- **NEW**: Batch prediction API endpoint optimized for processing large CSV files

## Citations
If you use the UNSW-NB15 dataset, please cite the authors as indicated by UNSW.
