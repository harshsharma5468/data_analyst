"""
Machine Learning component for the Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, mean_squared_error, 
                             mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import json
import io
import joblib
from datetime import datetime
from components.utils import get_column_types

def prepare_data_for_ml(df: pd.DataFrame, target_col: str, test_size: float = 0.2, 
                        random_state: int = 42, stratify: bool = False):
    """
    Prepare data for ML with safety checks for rare classes
    """
    # 1. Drop rows with missing values in target
    df = df.dropna(subset=[target_col])

    # 2. Safety Check: Remove classes with only 1 member if stratifying
    if stratify:
        counts = df[target_col].value_counts()
        rare_classes = counts[counts < 2].index.tolist()
        if rare_classes:
            st.warning(f"Removing {len(rare_classes)} rare classes to allow stratified splitting.")
            df = df[~df[target_col].isin(rare_classes)]

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify column types
    col_types = get_column_types(X)
    numeric_features = col_types['numeric']
    categorical_features = col_types['categorical'] + col_types['boolean']
    
    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y if stratify else None
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_classification_model(X_train, X_test, y_train, y_test, preprocessor, 
                              model_type: str, hyperparameters: dict = None, 
                              cv_folds: int = 5):
    # Select model
    if model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier(random_state=42)
    
    if hyperparameters:
        model.set_params(**hyperparameters)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # Proba for ROC/AUC
    y_pred_proba = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            y_pred_proba = pipeline.predict_proba(X_test)
            if len(np.unique(y_test)) == 2: # Binary
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        except: pass

    return {'model': pipeline, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'metrics': metrics, 'y_test': y_test}

# ... (Include your existing train_regression_model, plot_confusion_matrix, etc. here) ...

def render_ml():
    st.title("🤖 Machine Learning")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    # UI Layout
    task = st.radio("Select Analysis Type", ["Classification", "Regression"], horizontal=True)
    target_col = st.selectbox("Select Target Variable (What to predict)", df.columns.tolist())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Set Size %", 10, 50, 20) / 100
    with col2:
        stratify = st.checkbox("Balance Classes (Stratify)", value=(task == "Classification"))
    with col3:
        cv_folds = st.number_input("Cross-Validation Folds", 2, 10, 5)

    # Model Selection logic
    model_options = ["Logistic Regression", "Random Forest", "XGBoost"] if task == "Classification" else ["Linear Regression", "Ridge", "Random Forest Regressor"]
    model_type = st.selectbox("Select Model Algorithm", model_options)

    if st.button("🚀 Train & Evaluate"):
        try:
            X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_ml(
                df, target_col, test_size, 42, stratify
            )

            if task == "Classification":
                result = train_classification_model(X_train, X_test, y_train, y_test, preprocessor, model_type)
            else:
                # Add your train_regression_model call here
                pass

            # Display Results
            st.success("Model trained successfully!")
            
            # Show Metrics
            m_cols = st.columns(len(result['metrics']))
            for i, (m_name, m_val) in enumerate(result['metrics'].items()):
                m_cols[i].metric(m_name.upper(), f"{m_val:.3f}")

            # Download Options (Memory Buffers)
            st.divider()
            st.subheader("💾 Export Model")
            
            # Model Pickle Buffer
            model_buffer = io.BytesIO()
            joblib.dump(result['model'], model_buffer)
            
            st.download_button(
                label="Download Trained Model (.pkl)",
                data=model_buffer.getvalue(),
                file_name=f"{model_type}_model.pkl",
                mime="application/octet-stream"
            )

        except Exception as e:
            st.error(f"Error: {e}")