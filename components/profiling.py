"""
Data profiling component for the Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import os
from datetime import datetime

def render_profiling():
    """
    Render the profiling page
    """
    st.title("Data Profiling")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    st.subheader("Generate Profile Report")
    
    # Profiling options
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Report Title", "Data Profiling Report")
        explorative = st.checkbox("Explorative Analysis", value=True)
    
    with col2:
        minimal = st.checkbox("Minimal Report", value=False)
        dark_mode = st.checkbox("Dark Mode", value=False)
    
    # Generate report button
    if st.button("Generate Profile Report"):
        with st.spinner("Generating profile report... This may take a moment."):
            try:
                # Generate profile report with compatible parameters
                profile_kwargs = {
                    'title': title,
                    'explorative': explorative,
                    'minimal': minimal
                }
                
                # Only add dark_mode if it's supported by the version
                try:
                    # Try creating a minimal report first to check parameter compatibility
                    test_profile = ProfileReport(df.head(10), title=title, minimal=True)
                    # If this works, we can try with dark_mode
                    if dark_mode:
                        profile_kwargs['dark_mode'] = True
                except TypeError:
                    # dark_mode parameter not supported, continue without it
                    st.warning("Dark mode not supported in this version of ydata-profiling")
                
                # Generate the full profile report
                profile = ProfileReport(df, **profile_kwargs)
                
                # Display in Streamlit
                st.subheader("Profile Report")
                profile_html = profile.to_html()
                
                # Use components to display HTML
                components.html(profile_html, height=800, scrolling=True)
                
                # Option to export
                st.subheader("Export Report")
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"models/reports/profile_report_{timestamp}.html"
                
                if st.button("Export HTML Report"):
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(profile_html)
                    st.success(f"Report saved to {file_name}")
                    
            except Exception as e:
                st.error(f"Error generating profile report: {str(e)}")
                # Show full traceback for debugging
                import traceback
                st.text(traceback.format_exc())
    
    # Show basic info without full profiling
    else:
        st.info("Click 'Generate Profile Report' to create a comprehensive data profile.")
        st.subheader("Basic Information")
        
        # Shape
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        st.write(f"**Memory Usage:** {memory_usage:.2f} MB")
        
        # Column info
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
