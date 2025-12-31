"""
Sidebar component for the Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
from components.utils import load_data, infer_dtypes, handle_missing_values, generate_sample_data
import os
from pathlib import Path

def render_sidebar():
    """
    Render the sidebar with navigation and data upload controls
    """
    with st.sidebar:
        st.image("assets/logo.png", width=100)
        st.title("Personal AI Data Analyst")
        
        # Navigation
        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "Data Upload",
                "EDA",
                "Profiling",
                "Visualization",
                "SQL Lab",
                "Machine Learning",
                "Time Series",
                "Reports"
            ]
        )
        
        st.markdown("---")
        
        # Data upload section
        st.subheader("Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel/Parquet",
            type=["csv", "xlsx", "xls", "parquet"],
            key="file_uploader"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            # Save uploaded file
            file_path = f"data/uploads/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Determine file type
            if uploaded_file.name.endswith('.csv'):
                file_type = "csv"
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                file_type = "excel"
            elif uploaded_file.name.endswith('.parquet'):
                file_type = "parquet"
            else:
                file_type = "csv"  # default
            
            # Load data
            with st.spinner("Loading data..."):
                df = load_data(file_path, file_type)
                if not df.empty:
                    df = infer_dtypes(df)
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Load sample data button
        if st.button("Load Sample Data"):
            with st.spinner("Generating sample data..."):
                df = generate_sample_data(5000)
                st.session_state.df = df
                st.session_state.file_name = "sample_data.csv"
                st.success("Sample data loaded!")
        
        st.markdown("---")
        
        # Data handling options
        if 'df' in st.session_state:
            st.subheader("Data Handling")
            
            # Missing value strategy
            missing_strategy = st.selectbox(
                "Missing Values",
                ["drop", "fill_mean", "fill_median", "fill_mode"],
                index=0
            )
            
            if st.button("Apply Data Handling"):
                with st.spinner("Processing data..."):
                    df_processed = handle_missing_values(
                        st.session_state.df, 
                        strategy=missing_strategy
                    )
                    st.session_state.df = df_processed
                    st.success("Data processing complete!")
        
        # Session controls
        st.markdown("---")
        st.subheader("Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()
        
        with col2:
            if st.button("Export Session"):
                st.warning("Export functionality not implemented yet")
        
        # Configuration
        st.markdown("---")
        st.subheader("Configuration")
        
        # Theme selection
        theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0 if st.session_state.get('theme', 'light') == 'light' else 1
        )
        st.session_state.theme = theme
        
        # Random state
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=st.session_state.get('random_state', 42)
        )
        st.session_state.random_state = random_state
    
    return page
