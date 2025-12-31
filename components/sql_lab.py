import streamlit as st
import pandas as pd
import duckdb
import io
import re

def execute_sql_on_dataframe(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """
    Execute SQL query by registering the dataframe with a dynamic name
    """
    try:
        # 1. Get the actual filename from session state
        raw_name = st.session_state.get('file_name', 'df')
        
        # 2. Clean the name so SQL can read it (e.g., "data.csv" -> "data")
        # SQL table names cannot have dots or start with numbers easily
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.split('.')[0])
        
        # 3. Register the dataframe with DuckDB under that specific name
        duckdb.register(table_name, df)
        
        # Also register as 'df' as a backup/shortcut
        duckdb.register("df", df)

        # 4. Run the query
        result = duckdb.query(sql_query).to_df()
        return result
    except Exception as e:
        st.error(f"SQL Error: {str(e)}")
        return pd.DataFrame()

def render_sql_lab():
    st.title("SQL Lab")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.df
    
    # Identify the name the user should use in their SQL
    raw_name = st.session_state.get('file_name', 'df')
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.split('.')[0])
    
    st.subheader(f"Querying Dataset: `{raw_name}`")
    
    # Inform the user what the table is called
    st.info(f"🚀 Your table name is: **`{table_name}`** (or simply use **`df`**)")
    
    # SQL query input
    default_query = f"SELECT * FROM {table_name} LIMIT 10"
    sql_query = st.text_area(
        "Enter SQL Query",
        value=default_query,
        height=150
    )
    
    if st.button("Execute Query"):
        if sql_query.strip():
            with st.spinner("Executing..."):
                result_df = execute_sql_on_dataframe(df, sql_query)
                
                if not result_df.empty:
                    st.success(f"Query successful! Showing {len(result_df)} rows.")
                    st.dataframe(result_df)
                    
                    # CSV/Excel Download Buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download CSV", result_df.to_csv(index=False), "query.csv", "text/csv")
                    with col2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            result_df.to_excel(writer, index=False)
                        st.download_button("Download Excel", buffer.getvalue(), "query.xlsx")