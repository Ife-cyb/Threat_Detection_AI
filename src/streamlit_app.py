import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

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


st.title("Threat Detection AI Dashboard")

with st.sidebar:
    st.header("Model")
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
        st.success("Artifacts loaded")

st.markdown("Upload a CSV of network flows (columns like proto, service, state, sbytes, dbytes, spkts, dpkts, etc.)")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

example_note = st.expander("Expected columns (examples)")
with example_note:
    st.code("proto,service,state,source_ip,dest_ip,spkts,dpkts,sbytes,dbytes,ct_state_ttl,...")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
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

    st.subheader("Charts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Attack distribution (predicted)**")
        dist = result_df["prediction"].value_counts().rename({0: "Normal", 1: "Attack"})
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=dist.index, y=dist.values, ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Packet/Byte stats**")
        numeric_cols = [c for c in ["spkts", "dpkts", "sbytes", "dbytes"] if c in result_df.columns]
        if numeric_cols:
            agg = result_df.groupby("prediction")[numeric_cols].sum().rename(index={0: "Normal", 1: "Attack"})
            st.dataframe(agg)
        else:
            st.info("No packet/byte columns found in uploaded CSV")

    # If ground truth in upload, show confusion matrix
    if "label" in df.columns:
        st.subheader("Confusion Matrix (if ground truth 'label' present)")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df["label"].astype(int), preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Upload a CSV to get started.") 