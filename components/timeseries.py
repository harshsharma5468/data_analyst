"""
Time Series component for the Personal AI Data Analyst
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Parse and set date column as index
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_col (str): Date column name
        
    Returns:
        pd.DataFrame: Dataframe with date column as index
    """
    df_copy = df.copy()
    
    # Convert to datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Set as index
    df_copy = df_copy.set_index(date_col).sort_index()
    
    return df_copy

def infer_frequency(df: pd.DataFrame) -> str:
    """
    Infer frequency of time series data
    
    Args:
        df (pd.DataFrame): Time series dataframe with datetime index
        
    Returns:
        str: Inferred frequency
    """
    try:
        freq = pd.infer_freq(df.index)
        return freq if freq else "Unknown"
    except:
        return "Unknown"

def resample_data(df: pd.DataFrame, value_col: str, freq: str, agg_method: str = "sum") -> pd.DataFrame:
    """
    Resample time series data
    
    Args:
        df (pd.DataFrame): Input dataframe
        value_col (str): Column to resample
        freq (str): Frequency to resample to
        agg_method (str): Aggregation method
        
    Returns:
        pd.DataFrame: Resampled dataframe
    """
    if agg_method == "sum":
        return df[value_col].resample(freq).sum()
    elif agg_method == "mean":
        return df[value_col].resample(freq).mean()
    elif agg_method == "count":
        return df[value_col].resample(freq).count()
    else:
        return df[value_col].resample(freq).sum()

def decompose_time_series(series: pd.Series, model: str = "additive", period: int = None):
    """
    Decompose time series into trend, seasonal, and residual components
    
    Args:
        series (pd.Series): Time series data
        model (str): Model type ('additive' or 'multiplicative')
        period (int): Period of seasonality
        
    Returns:
        DecomposeResult: Decomposition result
    """
    if period is None:
        # Try to infer period
        if len(series) >= 24:
            period = 12  # Monthly data with yearly seasonality
        elif len(series) >= 30:
            period = 7   # Daily data with weekly seasonality
        else:
            period = min(12, len(series)//2)
    
    return seasonal_decompose(series, model=model, period=period)

def fit_sarimax_model(series: pd.Series, order: tuple = (1,1,1), seasonal_order: tuple = (1,1,1,12)):
    """
    Fit SARIMAX model to time series data
    
    Args:
        series (pd.Series): Time series data
        order (tuple): (p,d,q) order
        seasonal_order (tuple): (P,D,Q,s) seasonal order
        
    Returns:
        SARIMAXResults: Fitted model
    """
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    return fitted_model

def forecast_with_confidence(model, steps: int = 12):
    """
    Generate forecast with confidence intervals
    
    Args:
        model: Fitted time series model
        steps (int): Number of steps to forecast
        
    Returns:
        pd.DataFrame: Forecast with confidence intervals
    """
    forecast = model.get_forecast(steps=steps)
    forecast_df = forecast.conf_int()
    forecast_df['forecast'] = forecast.predicted_mean
    return forecast_df

def render_timeseries():
    """
    Render the Time Series page
    """
    st.title("Time Series Analysis")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    # Find date columns
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    if not date_cols:
        # Try to find potential date columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            try:
                pd.to_datetime(df[col].iloc[:10])  # Test first 10 values
                date_cols.append(col)
            except:
                pass
    
    if not date_cols:
        st.warning("No date columns found in the dataset. Please ensure your data has a date/datetime column.")
        return
    
    # Date column selection
    date_col = st.selectbox("Select Date Column", date_cols)
    
    # Parse date column
    df_ts = parse_date_column(df, date_col)
    
    # Show frequency information
    freq = infer_frequency(df_ts)
    st.info(f"Inferred frequency: {freq}")
    
    # Value column selection
    numeric_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for time series analysis.")
        return
    
    value_col = st.selectbox("Select Value Column", numeric_cols)
    
    # Resampling options
    st.subheader("Data Aggregation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        resample_freq = st.selectbox(
            "Resample Frequency",
            ["D", "W", "M", "Q", "Y"],
            format_func=lambda x: {
                "D": "Daily",
                "W": "Weekly", 
                "M": "Monthly",
                "Q": "Quarterly",
                "Y": "Yearly"
            }[x]
        )
    
    with col2:
        agg_method = st.selectbox(
            "Aggregation Method",
            ["sum", "mean", "count"]
        )
    
    # Resample data
    resampled_series = resample_data(df_ts, value_col, resample_freq, agg_method)
    
    # Show resampled data
    st.subheader("Resampled Time Series")
    st.line_chart(resampled_series)
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Decomposition", "Modeling", "Forecasting"])
    
    with tab1:
        st.subheader("Time Series Decomposition")
        
        model_type = st.selectbox("Model Type", ["additive", "multiplicative"])
        
        if st.button("Decompose Series"):
            with st.spinner("Decomposing time series..."):
                try:
                    decomposition = decompose_time_series(resampled_series, model_type)
                    
                    # Plot decomposition
                    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                    
                    decomposition.observed.plot(ax=axes[0])
                    axes[0].set_title('Original')
                    
                    decomposition.trend.plot(ax=axes[1])
                    axes[1].set_title('Trend')
                    
                    decomposition.seasonal.plot(ax=axes[2])
                    axes[2].set_title('Seasonal')
                    
                    decomposition.resid.plot(ax=axes[3])
                    axes[3].set_title('Residual')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error in decomposition: {str(e)}")
    
    with tab2:
        st.subheader("Time Series Modeling")
        
        # SARIMAX parameters
        st.write("SARIMAX Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            p = st.number_input("p (AR)", min_value=0, max_value=5, value=1)
            d = st.number_input("d (Integration)", min_value=0, max_value=2, value=1)
            q = st.number_input("q (MA)", min_value=0, max_value=5, value=1)
        
        with col2:
            P = st.number_input("P (Seasonal AR)", min_value=0, max_value=3, value=1)
            D = st.number_input("D (Seasonal Integration)", min_value=0, max_value=2, value=1)
            Q = st.number_input("Q (Seasonal MA)", min_value=0, max_value=3, value=1)
            s = st.number_input("s (Seasonal Period)", min_value=2, max_value=365, value=12)
        
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        
        if st.button("Fit SARIMAX Model"):
            with st.spinner("Fitting SARIMAX model..."):
                try:
                    model = fit_sarimax_model(resampled_series, order, seasonal_order)
                    
                    # Show model summary
                    st.subheader("Model Summary")
                    st.text(model.summary())
                    
                    # Store model in session state
                    st.session_state.ts_model = model
                    st.session_state.ts_series = resampled_series
                    
                    st.success("Model fitted successfully!")
                    
                except Exception as e:
                    st.error(f"Error fitting model: {str(e)}")
    
    with tab3:
        st.subheader("Forecasting")
        
        if 'ts_model' not in st.session_state:
            st.warning("Please fit a model first in the Modeling tab.")
            return
        
        steps = st.number_input("Forecast Steps", min_value=1, max_value=100, value=12)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    model = st.session_state.ts_model
                    forecast_df = forecast_with_confidence(model, steps)
                    
                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data
                    ax.plot(st.session_state.ts_series.index, st.session_state.ts_series.values, 
                            label='Historical', color='blue')
                    
                    # Plot forecast
                    ax.plot(forecast_df.index, forecast_df['forecast'], 
                            label='Forecast', color='red')
                    
                    # Plot confidence intervals
                    ax.fill_between(forecast_df.index, 
                                    forecast_df.iloc[:, 0], 
                                    forecast_df.iloc[:, 1], 
                                    color='pink', alpha=0.3, label='Confidence Interval')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_col)
                    ax.set_title('Time Series Forecast')
                    ax.legend()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show forecast table
                    st.subheader("Forecast Values")
                    st.dataframe(forecast_df)
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
