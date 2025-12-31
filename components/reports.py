import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from fpdf import FPDF

def render_reports():
    st.title("📊 Analysis Reports")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    # 1. Generate Report Content
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_content = f"""# Personal AI Data Analyst Report
Generated on: {timestamp_str}

## Data Overview
- Dataset: {st.session_state.get('file_name', 'Unknown')}
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Missing Values: {df.isnull().sum().sum()}

## Column Types
"""
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    report_content += f"- Numeric Columns ({len(num_cols)}): {', '.join(num_cols[:5])}...\n"
    report_content += f"- Categorical Columns ({len(cat_cols)}): {', '.join(cat_cols[:5])}...\n"

    if 'ml_metrics' in st.session_state:
        report_content += f"\n## Machine Learning Performance\n"
        report_content += f"- Task: {st.session_state.get('ml_task', 'N/A')}\n"
        report_content += f"- Metrics: {json.dumps(st.session_state['ml_metrics'], indent=2)}\n"

    st.markdown(report_content)

    # 2. Export Section
    st.subheader("📥 Export Report")
    col1, col2 = st.columns(2)
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        st.download_button(
            label="Download as Markdown",
            data=report_content,
            file_name=f"report_{file_ts}.md",
            mime="text/markdown"
        )

    with col2:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Personal AI Data Analyst Report", ln=True, align='C')
            pdf.set_font("Arial", "", 11)
            pdf.ln(10)
            
            # Sanitize text for PDF encoding (Latin-1)
            clean_text = report_content.replace("#", "").replace("**", "").encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 10, clean_text)
            
            # Fix: Convert bytearray to bytes for Streamlit
            pdf_bytes = bytes(pdf.output())
            
            st.download_button(
                label="Download as PDF",
                data=pdf_bytes,
                file_name=f"report_{file_ts}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF Export failed: {e}")

    # 3. Visualizations
    st.divider()
    st.subheader("Key Visualizations")
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)