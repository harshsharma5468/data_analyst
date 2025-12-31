"""
Exploratory Data Analysis component for the Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from components.utils import get_column_types, safe_plot
import io

@safe_plot
def show_data_summary(df: pd.DataFrame):
    """
    Show basic data summary
    """
    st.subheader("Data Summary")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())
    
    # Data types
    st.subheader("Data Types")
    dtype_counts = df.dtypes.value_counts()
    st.bar_chart(dtype_counts)
    
    # Column types breakdown
    col_types = get_column_types(df)
    for col_type, cols in col_types.items():
        if cols:
            with st.expander(f"{col_type.capitalize()} Columns ({len(cols)})"):
                st.write(cols)

@safe_plot
def show_missing_values(df: pd.DataFrame):
    """
    Show missing values matrix
    """
    st.subheader("Missing Values")
    
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title("Missing Values Matrix")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No missing values found in the dataset")

@safe_plot
def show_correlation_matrix(df: pd.DataFrame, method: str = "pearson"):
    """
    Show correlation matrix
    """
    st.subheader(f"Correlation Matrix ({method.capitalize()})")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title(f"{method.capitalize()} Correlation Matrix")
        st.pyplot(fig)
        plt.close()
        
        # Show strongest correlations
        st.subheader("Strongest Correlations")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Only show correlations > 0.5
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs).sort_values(
                'Correlation', 
                key=abs, 
                ascending=False
            )
            st.dataframe(corr_df)
        else:
            st.info("No strong correlations (>0.5) found")
    else:
        st.info("Not enough numeric columns for correlation analysis")

@safe_plot
def show_outliers(df: pd.DataFrame):
    """
    Show outlier detection using IQR method
    """
    st.subheader("Outlier Detection (IQR Method)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        outlier_info = []
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_info.append({
                    'Column': col,
                    'Outliers': outlier_count,
                    'Percentage': round((outlier_count / len(df)) * 100, 2)
                })
        
        if outlier_info:
            outlier_df = pd.DataFrame(outlier_info)
            st.dataframe(outlier_df)
            
            # Show boxplot for columns with outliers
            st.subheader("Boxplots for Columns with Outliers")
            cols_with_outliers = outlier_df['Column'].tolist()
            
            # Create subplots
            n_cols = min(len(cols_with_outliers), 4)
            n_rows = (len(cols_with_outliers) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(cols_with_outliers):
                df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'{col}')
            
            # Hide empty subplots
            for i in range(len(cols_with_outliers), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No outliers detected using IQR method")
    else:
        st.info("No numeric columns found for outlier detection")

@safe_plot
def show_distributions(df: pd.DataFrame):
    """
    Show feature distributions
    """
    st.subheader("Feature Distributions")
    
    col_types = get_column_types(df)
    
    # Numeric distributions
    if col_types['numeric']:
        st.subheader("Numeric Distributions")
        numeric_cols = col_types['numeric']
        
        # Create subplots
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Categorical distributions
    if col_types['categorical']:
        st.subheader("Categorical Distributions")
        categorical_cols = col_types['categorical']
        
        for col in categorical_cols[:5]:  # Limit to first 5 for performance
            with st.expander(f"Distribution of {col}"):
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                st.bar_chart(value_counts)

def render_eda():
    """
    Render the EDA page
    """
    st.title("Exploratory Data Analysis")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Tabs for different EDA components
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary", 
        "Missing Values", 
        "Correlations", 
        "Outliers", 
        "Distributions"
    ])
    
    with tab1:
        show_data_summary(df)
    
    with tab2:
        show_missing_values(df)
    
    with tab3:
        correlation_method = st.selectbox(
            "Correlation Method",
            ["pearson", "spearman"],
            index=0
        )
        show_correlation_matrix(df, correlation_method)
    
    with tab4:
        show_outliers(df)
    
    with tab5:
        show_distributions(df)
