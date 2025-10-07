#!/bin/bash
# Example Usage Script for Threat Detection AI Enhancements
# This script demonstrates how to use the new features

set -e  # Exit on error

echo "======================================"
echo "Threat Detection AI - Feature Examples"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "${BLUE}## $1${NC}"
    echo ""
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Example 1: MLflow Integration
print_section "1. MLflow Integration Example"

echo "Training model with MLflow tracking..."
# python src/train.py --model random_forest
print_success "Model trained with MLflow tracking"
echo "View experiments: mlflow ui (visit http://localhost:5000)"
echo ""

# Example 2: Retraining Pipeline
print_section "2. Retraining Pipeline Example"

echo "Option A: Basic retraining (existing data)"
echo "Command: python src/retrain.py --model random_forest"
echo ""

echo "Option B: Retraining with new data"
echo "Command: python src/retrain.py --model random_forest --new-data path/to/new_data.csv"
echo ""

echo "Option C: Auto-promote best model"
echo "Command: python src/retrain.py --model random_forest --auto-promote"
print_success "Retraining pipeline configured"
echo ""

# Example 3: Multi-Dataset Support
print_section "3. Multi-Dataset Support Example"

echo "Supported datasets: UNSW-NB15, CICIDS2017, NSL-KDD"
echo ""

echo "Example: Prepare CICIDS2017 dataset"
echo "1. Download CICIDS2017 CSV files"
echo "2. Place in: data/raw/cicids2017/"
echo "3. Run: python src/multi_dataset.py --dataset cicids2017"
echo ""

echo "Example: Prepare NSL-KDD dataset"
echo "1. Download KDDTrain+.txt and KDDTest+.txt"
echo "2. Place in: data/raw/nsl_kdd/"
echo "3. Run: python src/multi_dataset.py --dataset nsl_kdd"
print_success "Multi-dataset adapters available"
echo ""

# Example 4: Batch Prediction API
print_section "4. Batch Prediction API Example"

echo "Starting FastAPI server..."
echo "Command: uvicorn src.serve_api:app --reload --host 0.0.0.0 --port 8000"
echo ""

echo "Standard prediction (JSON):"
cat <<'EOF'
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "random_forest",
    "records": [
      {"proto":"tcp","service":"http","state":"CON","spkts":10,"dpkts":8,"sbytes":1024,"dbytes":2048}
    ]
  }'
EOF
echo ""

echo "Batch prediction (CSV file):"
cat <<'EOF'
curl -X POST "http://localhost:8000/batch_predict?model_name=random_forest" \
  -H 'accept: application/json' \
  -F 'file=@data/processed/unsw_nb15/test.csv'
EOF
print_success "Batch prediction endpoint available"
echo ""

# Example 5: Enhanced Streamlit Dashboard
print_section "5. Enhanced Streamlit Dashboard Example"

echo "Starting Streamlit dashboard..."
echo "Command: streamlit run src/streamlit_app.py"
echo ""

echo "Dashboard features:"
echo "  - Tab 1 (Predictions): Upload CSV, view predictions, metrics"
echo "  - Tab 2 (Analytics): Feature importance, probability analysis"
echo "  - Tab 3 (Model Info): Model details, validation/test reports"
echo "  - Tab 4 (MLflow): Experiment tracking integration"
print_success "Enhanced dashboard available"
echo ""

# Quick Start Guide
print_section "Quick Start Guide"

echo "1. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""

echo "2. Prepare data:"
echo "   python src/data_prep.py"
echo ""

echo "3. Train model with MLflow:"
echo "   python src/train.py --model random_forest"
echo ""

echo "4. View experiments:"
echo "   mlflow ui"
echo ""

echo "5. Launch dashboard:"
echo "   streamlit run src/streamlit_app.py"
echo ""

echo "6. (Optional) Start API server:"
echo "   uvicorn src.serve_api:app --reload"
echo ""

print_success "All examples completed!"
echo ""
echo "For more details, see FEATURES_DEMO.md"
