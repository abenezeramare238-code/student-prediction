import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = r"C:\Users\HP\Desktop\StudentPerformanceProject\student.csv"  # adjust if needed
MODEL_PATH = "student_rf.pkl"

st.set_page_config(page_title="Student Prediction", layout="centered")

st.title("Student Performance Predictor")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_or_train_model(df):
    # This cached resource will only attempt to load an existing model file.
    # Training is moved to an explicit action to avoid long startup delays.
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            # If model file is corrupted, remove it and return None
            st.warning(f"Model file corrupted ({type(e).__name__}). Removing and retraining needed.")
            os.remove(MODEL_PATH)
            return None
    return None


def train_and_save_model(df, n_estimators=100, max_rows=None):
    X = df[['weekly_self_study_hours','attendance_percentage','class_participation']]
    y = (df['total_score'] >= 50).astype(int)
    # Optionally sample to speed up quick training on very large datasets
    if max_rows is not None and len(df) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODEL_PATH)
    st.session_state['metrics'] = {"accuracy": accuracy_score(y_test, rf.predict(X_test))}
    st.session_state['model_trained'] = True  # Flag to indicate model was trained
    return rf

# Load
# Load
df = load_data(DATA_PATH)

# Initialize session state for model tracking
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# Try to load pre-trained model (will return None if not present)
if st.session_state['model_trained'] or os.path.exists(MODEL_PATH):
    model = load_or_train_model(df)
else:
    model = None

# Sidebar / inputs
st.sidebar.header("Student Input")
study_hours = st.sidebar.slider("Weekly study hours", 0.0, 60.0, 10.0, 0.5)
attendance = st.sidebar.slider("Attendance %", 0, 100, 80)
participation = st.sidebar.slider("Class participation (0-10)", 0, 10, 5)
# Model controls
st.sidebar.markdown("---")
if model is None:
    st.sidebar.info("No trained model found. Train a model or provide a pre-trained model file.")
    if st.sidebar.button("Train model now (quick)"):
        with st.spinner("Training model (this may take a while)..."):
            # train on a sample to make this quick; remove max_rows for full training
            model = train_and_save_model(df, n_estimators=100, max_rows=50000)
            st.success("Training completed and model saved.")
            st.rerun()
    if st.sidebar.button("Train full model"):
        with st.spinner("Training full model (may take several minutes)..."):
            model = train_and_save_model(df, n_estimators=200, max_rows=None)
            st.success("Full training completed and model saved.")
            st.rerun()
else:
    if st.sidebar.button("Reload model"):
        model = load_or_train_model(df)
        st.success("Model reloaded from disk.")

if st.sidebar.button("Predict"):
    if model is None:
        st.warning("No model available — please train or load a model first.")
    else:
        X_in = np.array([[study_hours, attendance, participation]])
        pred = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0].max() * 100
        st.success(f"Prediction: {'PASS' if pred==1 else 'FAIL'} — Confidence: {proba:.1f}%")

# Show data & metrics
with st.expander("Dataset preview"):
    st.dataframe(df.head())

if 'metrics' in st.session_state:
    st.markdown(f"**Model accuracy (test):** {st.session_state['metrics']['accuracy']:.3f}")

# Simple visualization
st.subheader("Feature distributions")
st.bar_chart(df[['weekly_self_study_hours','attendance_percentage','class_participation']].describe().loc['mean'])