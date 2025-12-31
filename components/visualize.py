"""
Visualization component for the Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from components.utils import get_column_types, safe_plot

def render_visualization():
    """
    Render the visualization page
    """
    st.title("Data Visualization")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    # Tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["Quick Charts", "Interactive Charts", "Custom Builder"])
    
    with tab1:
        render_quick_charts(df)
    
    with tab2:
        render_interactive_charts(df)
    
    with tab3:
        render_custom_builder(df)

@safe_plot
def render_quick_charts(df: pd.DataFrame):
    """
    Render quick charts
    """
    st.subheader("Quick Charts")
    
    col_types = get_column_types(df)
    
    # Histogram
    if col_types['numeric']:
        st.subheader("Histogram")
        numeric_col = st.selectbox("Select numeric column", col_types['numeric'])
        if numeric_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[numeric_col].dropna(), bins=30, edgecolor='black')
            ax.set_xlabel(numeric_col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {numeric_col}')
            st.pyplot(fig)
            plt.close()
    
    # Boxplot
    if col_types['numeric']:
        st.subheader("Boxplot")
        numeric_col = st.selectbox("Select numeric column for boxplot", col_types['numeric'], key="boxplot_col")
        if numeric_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(df[numeric_col].dropna())
            ax.set_ylabel(numeric_col)
            ax.set_title(f'Boxplot of {numeric_col}')
            st.pyplot(fig)
            plt.close()
    
    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        st.subheader("Correlation Heatmap")
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
        plt.close()

def render_interactive_charts(df: pd.DataFrame):
    """
    Render interactive charts with Plotly
    """
    st.subheader("Interactive Charts")
    
    col_types = get_column_types(df)
    
    chart_type = st.selectbox(
        "Chart Type",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Treemap", "Sunburst"]
    )
    
    if chart_type == "Scatter Plot":
        if len(col_types['numeric']) >= 2:
            x_col = st.selectbox("X-axis", col_types['numeric'], key="scatter_x")
            y_col = st.selectbox("Y-axis", col_types['numeric'], key="scatter_y")
            color_col = st.selectbox("Color", [None] + col_types['categorical'] + col_types['numeric'], key="scatter_color")
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Line Chart":
        if len(col_types['numeric']) >= 1 and (col_types['datetime'] or col_types['categorical']):
            x_col = st.selectbox("X-axis", col_types['datetime'] + col_types['categorical'], key="line_x")
            y_col = st.selectbox("Y-axis", col_types['numeric'], key="line_y")
            color_col = st.selectbox("Color", [None] + col_types['categorical'], key="line_color")
            
            if x_col and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Bar Chart":
        x_col = st.selectbox("X-axis", col_types['categorical'], key="bar_x")
        y_col = st.selectbox("Y-axis", col_types['numeric'], key="bar_y")
        color_col = st.selectbox("Color", [None] + col_types['categorical'], key="bar_color")
        
        if x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Treemap":
        if col_types['categorical']:
            path_cols = st.multiselect("Hierarchy Path", col_types['categorical'], key="treemap_path")
            values_col = st.selectbox("Values", col_types['numeric'], key="treemap_values")
            
            if path_cols and values_col:
                fig = px.treemap(df, path=path_cols, values=values_col, title="Treemap")
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Sunburst":
        if col_types['categorical']:
            path_cols = st.multiselect("Hierarchy Path", col_types['categorical'], key="sunburst_path")
            values_col = st.selectbox("Values", col_types['numeric'], key="sunburst_values")
            
            if path_cols and values_col:
                fig = px.sunburst(df, path=path_cols, values=values_col, title="Sunburst Chart")
                st.plotly_chart(fig, use_container_width=True)

def render_custom_builder(df: pd.DataFrame):
    """
    Render custom chart builder
    """
    st.subheader("Custom Chart Builder")
    
    col_types = get_column_types(df)
    
    # Chart type selection
    chart_type = st.selectbox(
        "Chart Type",
        ["Scatter", "Line", "Bar", "Histogram", "Box Plot", "Violin Plot"],
        key="custom_chart_type"
    )
    
    # X and Y axis selection
    if chart_type in ["Scatter", "Line", "Bar"]:
        x_col = st.selectbox("X-axis", df.columns.tolist(), key="custom_x")
        y_col = st.selectbox("Y-axis", df.columns.tolist(), key="custom_y")
    elif chart_type in ["Histogram", "Box Plot", "Violin Plot"]:
        x_col = st.selectbox("Column", col_types['numeric'], key="custom_single")
        y_col = None
    
    # Color/hue selection
    color_col = st.selectbox(
        "Color/Hue (optional)", 
        [None] + df.columns.tolist(), 
        key="custom_color"
    )
    
    # Facet selection
    facet_col = st.selectbox(
        "Facet Column (optional)", 
        [None] + col_types['categorical'], 
        key="custom_facet"
    )
    
    # Filters
    st.subheader("Filters")
    filter_cols = st.multiselect("Select columns to filter", df.columns.tolist())
    
    filters = {}
    for col in filter_cols:
        if df[col].dtype in ['object', 'category']:
            unique_vals = df[col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"Filter {col}", unique_vals, key=f"filter_{col}")
            if selected_vals:
                filters[col] = selected_vals
        else:
            min_val, max_val = df[col].min(), df[col].max()
            selected_range = st.slider(
                f"Filter {col}", 
                float(min_val), 
                float(max_val), 
                (float(min_val), float(max_val)),
                key=f"filter_{col}"
            )
            filters[col] = selected_range
    
    # Apply filters
    filtered_df = df.copy()
    for col, filter_val in filters.items():
        if df[col].dtype in ['object', 'category']:
            if filter_val:
                filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
        else:
            min_val, max_val = filter_val
            filtered_df = filtered_df[
                (filtered_df[col] >= min_val) & 
                (filtered_df[col] <= max_val)
            ]
    
    # Aggregation
    if chart_type in ["Line", "Bar"] and y_col:
        agg_method = st.selectbox(
            "Aggregation Method",
            ["mean", "sum", "count", "median", "min", "max"],
            key="custom_agg"
        )
    
    # Time grouping (if datetime column exists)
    datetime_cols = col_types['datetime']
    if datetime_cols and x_col in datetime_cols:
        time_group = st.selectbox(
            "Time Grouping",
            ["None", "Year", "Quarter", "Month", "Week", "Day"],
            key="custom_time_group"
        )
    
    # Generate chart button
    if st.button("Generate Chart"):
        if chart_type == "Scatter" and x_col and y_col:
            fig = px.scatter(
                filtered_df, 
                x=x_col, 
                y=y_col, 
                color=color_col,
                facet_col=facet_col
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Line" and x_col and y_col:
            if agg_method:
                # Group by x_col and aggregate y_col
                if color_col:
                    grouped_df = filtered_df.groupby([x_col, color_col])[y_col].agg(agg_method).reset_index()
                else:
                    grouped_df = filtered_df.groupby(x_col)[y_col].agg(agg_method).reset_index()
                
                fig = px.line(
                    grouped_df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    facet_col=facet_col,
                    title=f"{agg_method.capitalize()} of {y_col} over {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Bar" and x_col and y_col:
            if agg_method:
                # Group by x_col and aggregate y_col
                if color_col:
                    grouped_df = filtered_df.groupby([x_col, color_col])[y_col].agg(agg_method).reset_index()
                else:
                    grouped_df = filtered_df.groupby(x_col)[y_col].agg(agg_method).reset_index()
                
                fig = px.bar(
                    grouped_df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    facet_col=facet_col,
                    title=f"{agg_method.capitalize()} of {y_col} by {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Histogram" and x_col:
            fig = px.histogram(
                filtered_df, 
                x=x_col, 
                color=color_col,
                facet_col=facet_col,
                title=f"Distribution of {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot" and x_col:
            fig = px.box(
                filtered_df, 
                x=x_col if color_col else None, 
                y=x_col, 
                color=color_col,
                facet_col=facet_col,
                title=f"Box Plot of {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Violin Plot" and x_col:
            fig = px.violin(
                filtered_df, 
                x=x_col if color_col else None, 
                y=x_col, 
                color=color_col,
                facet_col=facet_col,
                title=f"Violin Plot of {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
