# Threat Detection AI (UNSW-NB15)

End-to-end intrusion detection using the UNSW-NB15 dataset: automated download, preprocessing, model training, evaluation, Streamlit dashboard, and FastAPI for integration.

## Features
- Automated dataset download (UNSW-NB15 via Kaggle or a public mirror), preprocessing, train/test split
- Robust preprocessing: categorical encoding (proto, service, state, etc.), numeric scaling
- Baseline models: RandomForest (default) and XGBoost
- Metrics: precision, recall, F1-score, confusion matrix (saved as image)
- Streamlit dashboard for CSV upload, predictions, probabilities, and charts
- Optional FastAPI `/predict` endpoint for programmatic inference

## Project Structure
```
threat-detection-ai/
├─ data/
│  ├─ raw/unsw_nb15/              # downloaded CSVs
│  └─ processed/unsw_nb15/        # processed train/test CSVs
├─ models/                        # preprocessor + trained models
├─ src/
│  ├─ data_prep.py                # download + preprocess
│  ├─ train.py                    # model training
│  ├─ evaluate.py                 # evaluation + confusion matrix
│  ├─ streamlit_app.py            # Streamlit dashboard
│  └─ serve_api.py                # FastAPI service
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
```
Artifacts:
- `models/model_<name>.joblib`
- `models/val_report_<name>.txt`

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
- Upload a CSV of network flows (columns like `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, ...)
- View predictions, threat probabilities, attack distribution, and packet/byte stats

6. Run the FastAPI service (optional):
```bash
uvicorn src.serve_api:app --reload --host 0.0.0.0 --port 8000
```
Example request:
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

## Notes on Datasets
- Default is UNSW-NB15 flow-based dataset. Kaggle download is the most reliable.
- If you have only the four-part files `UNSW-NB15_1.csv`..`UNSW-NB15_4.csv`, also include `NUSW-NB15_groundTruth.csv`; the script will merge labels by `id`.
- Target label is `label` (0: Normal, 1: Attack). If missing, it is derived from `attack_cat` (`Normal` vs others).

## Development Tips
- Preprocessor is saved separately to ensure consistent feature handling in apps and API.
- To extend: add feature selection, class-imbalance techniques (e.g., SMOTE), or advanced models.

## Citations
If you use the UNSW-NB15 dataset, please cite the authors as indicated by UNSW.
