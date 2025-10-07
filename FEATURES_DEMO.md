"""
Demo and Testing Guide for New Features
========================================

This file demonstrates the new features added to the Threat Detection AI system.

## 1. MLflow Integration

### Training with MLflow tracking:
```bash
# Train model with MLflow tracking (default)
python src/train.py --model random_forest

# Train without MLflow tracking
python src/train.py --model random_forest --no-mlflow

# View experiments in MLflow UI
mlflow ui
# Visit http://localhost:5000
```

MLflow tracks:
- Model parameters (n_estimators, max_depth, etc.)
- Performance metrics (accuracy, precision, recall, F1-score)
- Model artifacts
- Training runs with timestamps


## 2. Retraining Pipeline

### Basic retraining:
```bash
# Retrain existing model
python src/retrain.py --model random_forest

# Retrain with new labeled data
python src/retrain.py --model random_forest --new-data path/to/new_data.csv

# Auto-promote if new model performs better
python src/retrain.py --model random_forest --new-data new_data.csv --auto-promote

# Limit training dataset size (for memory constraints)
python src/retrain.py --model random_forest --max-samples 100000
```

Features:
- Compares new model vs old model performance
- Backs up old model before promotion
- Supports incremental learning with new data
- MLflow tracking for retraining experiments


## 3. Multi-Dataset Support

### Supported datasets:
- UNSW-NB15 (default)
- CICIDS2017
- NSL-KDD

### Using CICIDS2017:
```bash
# 1. Download CICIDS2017 CSV files manually
# Place them in: data/raw/cicids2017/

# 2. Prepare dataset
python src/multi_dataset.py --dataset cicids2017

# 3. Train model (update PROCESSED_DIR in train.py to data/processed/cicids2017)
python src/train.py --model random_forest
```

### Using NSL-KDD:
```bash
# 1. Download NSL-KDD files
# Place KDDTrain+.txt and KDDTest+.txt in: data/raw/nsl_kdd/

# 2. Prepare dataset
python src/multi_dataset.py --dataset nsl_kdd

# 3. Train model
python src/train.py --model random_forest
```


## 4. Batch Prediction API

### Standard prediction (small batches):
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "random_forest",
    "records": [
      {"proto":"tcp","service":"http","state":"CON","spkts":10,"dpkts":8,"sbytes":1024,"dbytes":2048}
    ]
  }'
```

### Batch prediction (large CSV files):
```bash
curl -X POST "http://localhost:8000/batch_predict?model_name=random_forest" \
  -H 'accept: application/json' \
  -F 'file=@/path/to/large_dataset.csv'
```

Response includes:
- predictions: List of predictions
- threat_probabilities: Probability scores
- total_records: Count of processed records
- total_threats: Count of detected threats


## 5. Enhanced Streamlit Dashboard

### New Features:

#### Tab 1: Predictions
- Upload CSV files
- View predictions with threat probabilities
- Summary metrics (total records, threats detected, threat rate)
- Download predictions as CSV
- Visualizations:
  - Attack distribution
  - Threat probability distribution
  - Network flow statistics
  - Confusion matrix (if ground truth available)
  - Classification metrics (accuracy, precision, recall, F1)
  - ROC curve (optional)

#### Tab 2: Analytics
- Feature importance charts (top 15 features)
- Threat probability analysis by ranges
- High-risk records detection (>80% probability)
- Probability range distribution

#### Tab 3: Model Info
- Current model details
- Validation report
- Test report
- Confusion matrix visualization

#### Tab 4: MLflow Experiments
- MLflow integration instructions
- Link to experiment tracking UI
- Quick start guide


## Testing Checklist

### Prerequisites:
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run data preprocessing: `python src/data_prep.py`

### MLflow Integration:
- [ ] Train model with MLflow: `python src/train.py --model random_forest`
- [ ] Check MLflow UI: `mlflow ui` (visit http://localhost:5000)
- [ ] Verify experiment tracking data

### Retraining Pipeline:
- [ ] Run basic retraining: `python src/retrain.py --model random_forest`
- [ ] Test with new data (if available)
- [ ] Verify model comparison and backup

### Multi-Dataset Support:
- [ ] Test dataset adapter imports
- [ ] Prepare additional dataset (if files available)
- [ ] Verify unified preprocessing

### Batch Prediction:
- [ ] Start FastAPI: `uvicorn src.serve_api:app --reload`
- [ ] Test standard endpoint: `/predict`
- [ ] Test batch endpoint: `/batch_predict` with CSV file
- [ ] Verify response format

### Enhanced Dashboard:
- [ ] Start Streamlit: `streamlit run src/streamlit_app.py`
- [ ] Upload test CSV
- [ ] Check all 4 tabs:
  - [ ] Predictions tab (visualizations)
  - [ ] Analytics tab (feature importance)
  - [ ] Model Info tab (reports)
  - [ ] MLflow tab (instructions)
- [ ] Test batch processing indicator
- [ ] Download predictions CSV


## Performance Improvements

### Batch Prediction Optimization:
- Uses pandas vectorization for efficient processing
- Single preprocessing and prediction pass
- Optimized for large datasets (1000+ records)
- Reduced overhead compared to individual predictions

### Dashboard Enhancements:
- Cached model loading (Streamlit cache_resource)
- Efficient data processing with pandas
- Interactive visualizations with matplotlib/seaborn
- Tab-based organization for better UX


## Common Issues & Solutions

### Issue: MLflow not tracking
Solution: Ensure MLflow is installed (`pip install mlflow`) and not disabled with `--no-mlflow`

### Issue: Multi-dataset files not found
Solution: Download dataset files and place in correct directory (data/raw/{dataset_name}/)

### Issue: Batch prediction fails
Solution: Ensure CSV has compatible columns, check for missing preprocessing artifacts

### Issue: Feature importance not showing
Solution: Feature importance only available for tree-based models (RandomForest, XGBoost)

### Issue: Dashboard tabs not loading
Solution: Upload data in Predictions tab first, then view Analytics tab
"""