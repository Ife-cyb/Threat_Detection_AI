import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

MODELS_DIR = Path("models")

st.set_page_config(page_title="Threat Detection AI", layout="wide")

@st.cache_resource
def load_artifacts(model_name: str):
    try:
        artifacts = joblib.load(MODELS_DIR / "preprocessor.joblib")
        preprocessor = artifacts["preprocessor"]
    except FileNotFoundError:
        return None, None, "Preprocessor not found. Run preprocessing first."
    try:
        model = joblib.load(MODELS_DIR / f"model_{model_name}.joblib")
    except FileNotFoundError:
        return preprocessor, None, f"Model '{model_name}' not found. Train the model first."
    return preprocessor, model, None


def infer(df: pd.DataFrame, preprocessor, model):
    feature_df = df.copy()
    if "label" in feature_df.columns:
        feature_df = feature_df.drop(columns=["label"])  # optional ground truth
    X = preprocessor.transform(feature_df)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds, probs


def get_feature_importance(model, preprocessor):
    """Extract feature importance if available."""
    if hasattr(model, 'feature_importances_'):
        # Get feature names from preprocessor
        feature_names = []
        if hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    if hasattr(trans, 'get_feature_names_out'):
                        feature_names.extend(trans.get_feature_names_out(cols))
        
        if len(feature_names) == len(model.feature_importances_):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
    return None


st.title("Threat Detection AI Dashboard")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_name = st.selectbox("Choose model", ["random_forest", "xgboost"], index=0)
    preprocessor, model, load_err = load_artifacts(model_name)
    
    if load_err:
        st.error(load_err)
        st.info(
            "Generate artifacts with:\n"
            "1) Preprocess (Kaggle):\n"
            "   python src/data_prep.py --use-kaggle --kaggle-dataset galaxy-dl/unsw-nb15 --kaggle-files UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv\n"
            "   or mirror: python src/data_prep.py\n"
            "2) Train: python src/train.py --model " + model_name
        )
        st.stop()
    else:
        st.success(f"✓ Model loaded: {model_name}")
    
    st.divider()
    
    # Analytics options
    st.header("Analytics Options")
    show_feature_importance = st.checkbox("Show Feature Importance", value=True)
    show_advanced_metrics = st.checkbox("Show Advanced Metrics", value=False)
    batch_size_threshold = st.slider("Batch processing threshold", 100, 10000, 1000, 100,
                                     help="Use batch processing for files with more rows")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Analytics", "🔍 Model Info", "📚 MLflow Experiments"])

with tab1:
    st.markdown("Upload a CSV of network flows (columns like proto, service, state, sbytes, dbytes, spkts, dpkts, etc.)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_tab1")

    example_note = st.expander("Expected columns (examples)")
    with example_note:
        st.code("proto,service,state,source_ip,dest_ip,spkts,dpkts,sbytes,dbytes,ct_state_ttl,...")

    if uploaded:
        df = pd.read_csv(uploaded)
        
        # Batch processing indicator
        is_batch = len(df) > batch_size_threshold
        if is_batch:
            st.info(f"📦 Processing {len(df)} records in batch mode...")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(50))

        preds, probs = infer(df, preprocessor, model)
        result_df = df.copy()
        result_df["prediction"] = preds
        if probs is not None:
            result_df["threat_probability"] = probs

        st.subheader("Predictions")
        st.dataframe(result_df.head(100))

        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False)
        st.download_button("Download Predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(result_df))
        with col2:
            threats = (result_df["prediction"] == 1).sum()
            st.metric("Threats Detected", threats)
        with col3:
            threat_pct = (threats / len(result_df)) * 100 if len(result_df) > 0 else 0
            st.metric("Threat Rate", f"{threat_pct:.2f}%")
        with col4:
            if probs is not None:
                avg_threat_prob = result_df[result_df["prediction"] == 1]["threat_probability"].mean()
                st.metric("Avg Threat Probability", f"{avg_threat_prob:.2%}" if not pd.isna(avg_threat_prob) else "N/A")

        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Attack Distribution**")
            dist = result_df["prediction"].value_counts().rename({0: "Normal", 1: "Attack"})
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=dist.index, y=dist.values, ax=ax, palette=["green", "red"])
            ax.set_ylabel("Count")
            ax.set_xlabel("Classification")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            if probs is not None:
                st.markdown("**Threat Probability Distribution**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(result_df["threat_probability"], bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel("Threat Probability")
                ax.set_ylabel("Frequency")
                ax.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        # Packet/Byte statistics
        st.subheader("Network Flow Statistics")
        numeric_cols = [c for c in ["spkts", "dpkts", "sbytes", "dbytes"] if c in result_df.columns]
        if numeric_cols:
            agg = result_df.groupby("prediction")[numeric_cols].agg(['mean', 'sum']).round(2)
            agg.index = agg.index.map({0: "Normal", 1: "Attack"})
            st.dataframe(agg)
        else:
            st.info("No packet/byte columns found in uploaded CSV")

        # Ground truth comparison
        if "label" in df.columns:
            st.subheader("Performance Metrics (Ground Truth Available)")
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
            
            y_true = df["label"].astype(int)
            y_pred = preds
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                           xticklabels=["Normal", "Attack"], 
                           yticklabels=["Normal", "Attack"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.markdown("**Classification Metrics**")
                metrics = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, average='binary'),
                    "Recall": recall_score(y_true, y_pred, average='binary'),
                    "F1 Score": f1_score(y_true, y_pred, average='binary'),
                }
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.4f}")
                
                if show_advanced_metrics and probs is not None:
                    st.markdown("**ROC Curve**")
                    fpr, tpr, _ = roc_curve(y_true, probs)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    plt.close(fig)
    else:
        st.info("Upload a CSV to get started.")

with tab2:
    st.header("Analytics Dashboard")
    
    if uploaded and 'result_df' in locals():
        # Feature importance
        if show_feature_importance:
            st.subheader("Feature Importance")
            importance_df = get_feature_importance(model, preprocessor)
            if importance_df is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
                ax.set_title("Top 15 Most Important Features")
                st.pyplot(fig)
                plt.close(fig)
                
                with st.expander("View Feature Importance Table"):
                    st.dataframe(importance_df)
            else:
                st.info("Feature importance not available for this model type.")
        
        # Threat probability analysis
        if probs is not None:
            st.subheader("Threat Probability Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Probability Ranges**")
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
                result_df['prob_range'] = pd.cut(result_df['threat_probability'], bins=bins, labels=labels)
                range_counts = result_df['prob_range'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(6, 4))
                range_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_xlabel('Probability Range')
                ax.set_ylabel('Count')
                ax.set_title('Distribution by Probability Range')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.markdown("**High-Risk Records**")
                high_risk = result_df[result_df['threat_probability'] > 0.8]
                st.metric("High-Risk Count (>80%)", len(high_risk))
                
                if len(high_risk) > 0:
                    st.dataframe(high_risk.head(10))
    else:
        st.info("Upload data in the Predictions tab to view analytics.")

with tab3:
    st.header("Model Information")
    
    st.subheader("Current Model")
    st.write(f"**Model Type:** {model_name}")
    st.write(f"**Model Class:** {type(model).__name__}")
    
    # Try to load validation report
    val_report_path = MODELS_DIR / f"val_report_{model_name}.txt"
    if val_report_path.exists():
        st.subheader("Validation Report")
        with open(val_report_path, 'r') as f:
            st.text(f.read())
    
    # Try to load test report
    test_report_path = MODELS_DIR / f"test_report_{model_name}.txt"
    if test_report_path.exists():
        st.subheader("Test Report")
        with open(test_report_path, 'r') as f:
            st.text(f.read())
    
    # Show confusion matrix if available
    cm_path = MODELS_DIR / f"confusion_matrix_{model_name}.png"
    if cm_path.exists():
        st.subheader("Test Set Confusion Matrix")
        st.image(str(cm_path))

with tab4:
    st.header("MLflow Experiments")
    
    if HAS_MLFLOW:
        st.info("MLflow tracking is available. View experiments at: http://localhost:5000")
        st.markdown("""
        To view MLflow UI, run:
        ```bash
        mlflow ui
        ```
        Then visit http://localhost:5000 in your browser.
        """)
    else:
        st.warning("MLflow is not installed. Install it with: pip install mlflow")
    
    st.markdown("""
    ### Features:
    - Track model training experiments
    - Compare model performance across runs
    - Version control for models
    - Parameter and metric tracking
    
    ### Quick Start:
    1. Train a model: `python src/train.py --model random_forest`
    2. Launch MLflow UI: `mlflow ui`
    3. View experiments at http://localhost:5000
    """) 