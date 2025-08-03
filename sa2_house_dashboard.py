
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "SA2 Scores July 2025.xlsx"
    df = pd.read_excel(file_path, sheet_name="House", header=None)

    # Find the header row dynamically
    header_row_index = df[df.iloc[:, 1] == "SA2"].index[0]
    df.columns = df.iloc[header_row_index]
    df = df.drop(index=range(header_row_index + 1)).reset_index(drop=True)

    return df

df = load_data()

st.title("SA2 House Dashboard")

# Sidebar Filters
with st.sidebar:
    st.header("Filters")
    state_col = "State"
    type_col = "Property\nType"

    if state_col in df.columns and type_col in df.columns:
        states = st.multiselect("Select State(s):", sorted(df[state_col].dropna().unique()))
        property_types = st.multiselect("Select Property Type(s):", sorted(df[type_col].dropna().unique()))

        # Apply Filters
        if states:
            df = df[df[state_col].isin(states)]
        if property_types:
            df = df[df[type_col].isin(property_types)]
    else:
        st.error("Required columns not found in the dataset.")

# Display Data
st.dataframe(df, use_container_width=True)
