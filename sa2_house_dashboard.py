
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "SA2 Scores July 2025.xlsx"
    df = pd.read_excel(file_path, sheet_name="House", header=None)

    # Identify the header row dynamically where 'SA2' appears in the second column
    header_row_index = df[df.iloc[:, 1] == "SA2"].index[0]
    df.columns = df.iloc[header_row_index]
    df = df.drop(index=range(header_row_index + 1)).reset_index(drop=True)

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how='all')

    return df

df = load_data()

st.title("SA2 House Dashboard")

# Validate required columns
state_column = "State"
type_column = "Property\nType"

if state_column in df.columns and type_column in df.columns:
    with st.sidebar:
        st.header("Filters")
        selected_states = st.multiselect("Select State(s):", sorted(df[state_column].dropna().unique()))
        selected_types = st.multiselect("Select Property Type(s):", sorted(df[type_column].dropna().unique()))

    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df[state_column].isin(selected_states)]
    if selected_types:
        filtered_df = filtered_df[filtered_df[type_column].isin(selected_types)]

    st.dataframe(filtered_df, use_container_width=True)
else:
    missing = []
    if state_column not in df.columns:
        missing.append("'State'")
    if type_column not in df.columns:
        missing.append("'Property\\nType'")
    st.error("Missing required column(s): " + ", ".join(missing))
