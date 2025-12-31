"""
Main application file for Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
import yaml
import os
from pathlib import Path

# Import components
from components.sidebar import render_sidebar
from components.eda import render_eda
from components.profiling import render_profiling
from components.visualize import render_visualization
from components.sql_lab import render_sql_lab
from components.ml import render_ml
from components.timeseries import render_timeseries
from components.reports import render_reports
from components.utils import generate_sample_data

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file not found
        return {
            "theme": "light",
            "sample_data_path": "data/sample.csv",
            "random_state": 42
        }

def main():
    """Main application function"""
    # Set page config
    st.set_page_config(
        page_title="Personal AI Data Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load configuration
    config = load_config()
    
    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state.theme = config.get('theme', 'light')
    
    if 'random_state' not in st.session_state:
        st.session_state.random_state = config.get('random_state', 42)
    
    # Load sample data on first run if no data is loaded
    if 'df' not in st.session_state:
        sample_path = config.get('sample_data_path', 'data/sample.csv')
        if os.path.exists(sample_path):
            try:
                st.session_state.df = pd.read_csv(sample_path)
                st.session_state.file_name = "sample_data.csv"
            except:
                # Generate sample data if file doesn't exist
                st.session_state.df = generate_sample_data(1000)
                st.session_state.file_name = "generated_sample.csv"
        else:
            # Generate sample data if file doesn't exist
            st.session_state.df = generate_sample_data(1000)
            st.session_state.file_name = "generated_sample.csv"
    
    # Apply custom CSS
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Data Upload":
        st.title("Data Upload")
        st.info("Use the sidebar to upload data or load sample data.")
        if 'df' in st.session_state:
            st.subheader("Current Data")
            st.dataframe(st.session_state.df.head())
            st.write(f"Shape: {st.session_state.df.shape}")
    
    elif page == "EDA":
        render_eda()
    
    elif page == "Profiling":
        render_profiling()
    
    elif page == "Visualization":
        render_visualization()
    
    elif page == "SQL Lab":
        render_sql_lab()
    
    elif page == "Machine Learning":
        render_ml()
    
    elif page == "Time Series":
        render_timeseries()
    
    elif page == "Reports":
        render_reports()

if __name__ == "__main__":
    main()
