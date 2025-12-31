"""
Utility functions for the Personal AI Data Analyst
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Any, Dict, List
import io
import chardet
from pathlib import Path

# Create necessary directories
Path("data/uploads").mkdir(parents=True, exist_ok=True)
Path("models/saved").mkdir(parents=True, exist_ok=True)
Path("models/reports").mkdir(parents=True, exist_ok=True)

@st.cache_data
def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Detected encoding
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']

@st.cache_data
def load_data(file_path: str, file_type: str = "csv") -> pd.DataFrame:
    """
    Load data from various file formats with automatic encoding detection
    
    Args:
        file_path (str): Path to the file
        file_type (str): Type of file (csv, excel, parquet)
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        if file_type == "csv":
            encoding = detect_encoding(file_path)
            # Try to detect delimiter
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                if ';' in first_line:
                    delimiter = ';'
                elif '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','
            
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        elif file_type == "excel":
            df = pd.read_excel(file_path)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and optimize data types for a DataFrame
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with optimized dtypes
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        col_data = df_copy[col]
        
        # Skip if column is empty
        if col_data.isnull().all():
            continue
            
        # Try to convert to numeric
        if col_data.dtype == 'object':
            # Try to convert to numeric
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            if not numeric_data.isnull().all():
                # Check if it's actually integer
                if (numeric_data.dropna() % 1 == 0).all():
                    df_copy[col] = numeric_data.astype('Int64')
                else:
                    df_copy[col] = numeric_data
                    
            # Try to convert to datetime
            else:
                try:
                    df_copy[col] = pd.to_datetime(col_data)
                except:
                    pass
                    
        # Optimize numeric types
        elif col_data.dtype in ['int64', 'int32', 'int16', 'int8']:
            col_min, col_max = col_data.min(), col_data.max()
            if col_min >= -128 and col_max <= 127:
                df_copy[col] = col_data.astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df_copy[col] = col_data.astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df_copy[col] = col_data.astype('int32')
                
        elif col_data.dtype in ['float64', 'float32']:
            col_min, col_max = col_data.min(), col_data.max()
            if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                df_copy[col] = col_data.astype('float32')
                
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in a DataFrame
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode')
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_copy = df.copy()
    
    if strategy == "drop":
        df_copy = df_copy.dropna()
    elif strategy == "fill_mean":
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    elif strategy == "fill_median":
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    elif strategy == "fill_mode":
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else df_copy[col].fillna(method='ffill'))
            
    return df_copy

def safe_plot(func):
    """
    Decorator to safely handle plotting functions
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.warning(f"Plotting error: {str(e)}")
            return None
    return wrapper

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get column types from a DataFrame
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, List[str]]: Dictionary with column types
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'boolean': boolean_cols
    }

def generate_sample_data(n_rows: int = 5000) -> pd.DataFrame:
    """
    Generate sample retail data
    
    Args:
        n_rows (int): Number of rows to generate
        
    Returns:
        pd.DataFrame: Sample retail data
    """
    np.random.seed(42)
    
    # Generate dates over 18 months
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2024-06-30')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Beauty', 'Toys', 'Automotive']
    
    # Cities
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']
    
    # Payment methods
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'PayPal', 'Apple Pay']
    
    # Generate data
    data = []
    for _ in range(n_rows):
        date = np.random.choice(dates)
        customer_id = np.random.randint(1000, 9999)
        product_category = np.random.choice(categories)
        city = np.random.choice(cities)
        quantity = np.random.randint(1, 10)
        unit_price = np.round(np.random.uniform(5, 500), 2)
        discount = np.round(np.random.uniform(0, 0.3), 2)
        payment_method = np.random.choice(payment_methods)
        
        revenue = quantity * unit_price * (1 - discount)
        
        data.append({
            'date': date,
            'customer_id': customer_id,
            'product_category': product_category,
            'city': city,
            'quantity': quantity,
            'unit_price': unit_price,
            'discount': discount,
            'payment_method': payment_method,
            'revenue': revenue
        })
    
    df = pd.DataFrame(data)
    return df.sort_values('date').reset_index(drop=True)
