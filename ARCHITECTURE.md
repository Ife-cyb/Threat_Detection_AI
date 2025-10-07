# Threat Detection AI - System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THREAT DETECTION AI SYSTEM                            │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA LAYER                                   │    │
│  │                                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │  UNSW-NB15   │  │  CICIDS2017  │  │   NSL-KDD    │              │    │
│  │  │   (default)  │  │    (new)     │  │    (new)     │              │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │    │
│  │         │                  │                  │                      │    │
│  │         └──────────────────┼──────────────────┘                      │    │
│  │                            ▼                                         │    │
│  │                  ┌─────────────────────┐                            │    │
│  │                  │  Multi-Dataset      │                            │    │
│  │                  │  Adapter System     │  ← multi_dataset.py        │    │
│  │                  │  (NEW)              │                            │    │
│  │                  └──────────┬──────────┘                            │    │
│  │                             │                                        │    │
│  │                             ▼                                        │    │
│  │                  ┌─────────────────────┐                            │    │
│  │                  │  Preprocessor       │                            │    │
│  │                  │  (StandardScaler +  │  ← data_prep.py            │    │
│  │                  │   OneHotEncoder)    │                            │    │
│  │                  └──────────┬──────────┘                            │    │
│  │                             │                                        │    │
│  │                             ▼                                        │    │
│  │            ┌────────────────────────────────────┐                   │    │
│  │            │  Processed Data (train.csv/test.csv) │                 │    │
│  │            └────────────────────────────────────┘                   │    │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     TRAINING & EXPERIMENTATION                       │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    MLflow Tracking (NEW)                     │    │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │    │    │
│  │  │  │ Parameters │  │  Metrics   │  │  Model Artifacts   │    │    │    │
│  │  │  │  Logging   │  │  Logging   │  │    Versioning      │    │    │    │
│  │  │  └────────────┘  └────────────┘  └────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                             │                                         │    │
│  │                             ▼                                         │    │
│  │         ┌──────────────────────────────────────────┐                 │    │
│  │         │         Training Pipeline                 │                │    │
│  │         │                                            │                │    │
│  │         │  ┌────────────┐      ┌────────────────┐  │                │    │
│  │         │  │  train.py  │      │  retrain.py    │  │                │    │
│  │         │  │  (Enhanced)│      │  (NEW)         │  │                │    │
│  │         │  │            │      │                │  │                │    │
│  │         │  │ • MLflow   │      │ • Incremental  │  │                │    │
│  │         │  │   tracking │      │   learning     │  │                │    │
│  │         │  │ • Model    │      │ • Model        │  │                │    │
│  │         │  │   training │      │   comparison   │  │                │    │
│  │         │  └────────────┘      │ • Auto-promote │  │                │    │
│  │         │                       └────────────────┘  │                │    │
│  │         └──────────────────────────────────────────┘                 │    │
│  │                             │                                         │    │
│  │                             ▼                                         │    │
│  │                  ┌─────────────────────┐                             │    │
│  │                  │  Trained Models     │                             │    │
│  │                  │  • RandomForest     │                             │    │
│  │                  │  • XGBoost          │                             │    │
│  │                  └─────────────────────┘                             │    │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      INFERENCE LAYER                                 │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │               FastAPI Service (Enhanced)                     │    │    │
│  │  │                                                               │    │    │
│  │  │  ┌───────────────────────┐  ┌──────────────────────────┐    │    │    │
│  │  │  │  /predict             │  │  /batch_predict (NEW)    │    │    │    │
│  │  │  │                       │  │                          │    │    │    │
│  │  │  │  • JSON input         │  │  • CSV file upload       │    │    │    │
│  │  │  │  • Single/small batch │  │  • Optimized batching    │    │    │    │
│  │  │  │  • Real-time response │  │  • Large-scale inference │    │    │    │
│  │  │  └───────────────────────┘  └──────────────────────────┘    │    │    │
│  │  │                                                               │    │    │
│  │  └───────────────────────────────────────────────────────────────┘    │  │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │          Streamlit Dashboard (Major Enhancement)             │    │    │
│  │  │                                                               │    │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │    │
│  │  │  │   Tab 1  │  │   Tab 2  │  │   Tab 3  │  │   Tab 4  │    │    │    │
│  │  │  │          │  │          │  │          │  │          │    │    │    │
│  │  │  │Prediction│  │Analytics │  │Model Info│  │  MLflow  │    │    │    │
│  │  │  │          │  │          │  │          │  │          │    │    │    │
│  │  │  │• Upload  │  │• Feature │  │• Reports │  │• Exp.    │    │    │    │
│  │  │  │  CSV     │  │  import. │  │• Metrics │  │  tracking│    │    │    │
│  │  │  │• Metrics │  │• ROC     │  │• Conf.   │  │• Guide   │    │    │    │
│  │  │  │• Charts  │  │  curves  │  │  matrix  │  │          │    │    │    │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │    │
│  │  │                                                               │    │    │
│  │  └───────────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      MONITORING & TRACKING                           │    │
│  │                                                                       │    │
│  │                     ┌──────────────────────┐                         │    │
│  │                     │   MLflow UI          │                         │    │
│  │                     │   (http://localhost:5000)                      │    │
│  │                     │                      │                         │    │
│  │                     │  • Experiment runs   │                         │    │
│  │                     │  • Model comparison  │                         │    │
│  │                     │  • Metric tracking   │                         │    │
│  │                     │  • Model registry    │                         │    │
│  │                     └──────────────────────┘                         │    │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

                            DATA FLOW DIAGRAM
                            ─────────────────

    Raw Data → Dataset Adapter → Preprocessor → Processed Data
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │  Training      │
                                              │  Pipeline      │
                                              │                │
                                              │  • train.py    │
                                              │  • retrain.py  │
                                              │                │
                                              └────────┬───────┘
                                                       │
                                        ┌──────────────┴──────────────┐
                                        │                             │
                                        ▼                             ▼
                                   MLflow Tracking              Trained Model
                                   (Experiments)                (model.joblib)
                                                                      │
                                                      ┌───────────────┴───────────────┐
                                                      │                               │
                                                      ▼                               ▼
                                                FastAPI Service               Streamlit Dashboard
                                                (Inference)                   (Visualization)
                                                      │                               │
                                                      ▼                               ▼
                                                Predictions                      Analytics
                                                (JSON/CSV)                       (Interactive)


                        KEY ENHANCEMENTS IMPLEMENTED
                        ────────────────────────────

    1. ✅ MLflow Integration
       - Experiment tracking
       - Model versioning
       - Metrics logging

    2. ✅ Retraining Pipeline
       - Automated retraining
       - Model comparison
       - Auto-promotion

    3. ✅ Multi-Dataset Support
       - CICIDS2017 adapter
       - NSL-KDD adapter
       - Unified interface

    4. ✅ Batch Prediction
       - CSV file upload
       - Optimized processing
       - Large-scale inference

    5. ✅ Enhanced Dashboard
       - 4 interactive tabs
       - Feature importance
       - ROC curves
       - Advanced analytics
```

## Component Details

### Data Processing
- **data_prep.py**: Original UNSW-NB15 preprocessing
- **multi_dataset.py**: NEW - Multi-dataset adapter system
- Supports UNSW-NB15, CICIDS2017, NSL-KDD

### Training
- **train.py**: Enhanced with MLflow tracking
- **retrain.py**: NEW - Automated retraining pipeline
- Models: RandomForest, XGBoost

### Inference
- **serve_api.py**: Enhanced with batch prediction endpoint
- **streamlit_app.py**: Major upgrade with 4 tabs

### Monitoring
- **MLflow**: Experiment tracking and model registry
- Metrics: accuracy, precision, recall, F1-score
- Parameter logging and artifact versioning

## Usage Flow

```
1. Data Preparation
   ├── Download dataset (UNSW-NB15, CICIDS2017, or NSL-KDD)
   ├── Run preprocessing (data_prep.py or multi_dataset.py)
   └── Generate train/test CSV files

2. Model Training
   ├── Train model (train.py)
   ├── MLflow tracks experiment
   ├── Evaluate on test set (evaluate.py)
   └── Save model and reports

3. Model Deployment
   ├── Option A: FastAPI (serve_api.py)
   │   ├── /predict endpoint (JSON)
   │   └── /batch_predict endpoint (CSV)
   └── Option B: Streamlit (streamlit_app.py)
       ├── Upload CSV
       ├── View predictions
       ├── Analyze results
       └── Download predictions

4. Model Maintenance
   ├── Retrain with new data (retrain.py)
   ├── Compare performance
   ├── Auto-promote if better
   └── Track in MLflow
```
