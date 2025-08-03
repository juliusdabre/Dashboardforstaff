
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "SA2 Scores July 2025.xlsx"
    df = pd.read_excel(file_path, sheet_name="House", header=None)

    # Locate header row dynamically
    header_row_index = df[df.iloc[:, 1] == "SA2"].index[0]
    df.columns = df.iloc[header_row_index]
    df = df.drop(index=list(range(header_row_index + 1))).reset_index(drop=True)

    return df

df = load_data()

st.title("SA2 House Dashboard")

# Check for required columns
required_columns = ["State", "Property\nType"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"The following required column(s) are missing: {', '.join(missing_columns)}")
else:
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        selected_states = st.multiselect("Select State(s):", sorted(df["State"].dropna().unique()))
        selected_types = st.multiselect("Select Property Type(s):", sorted(df["Property\nType"].dropna().unique()))

    # Apply filters
    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]
    if selected_types:
        filtered_df = filtered_df[filtered_df["Property\nType"].isin(selected_types)]

    st.dataframe(filtered_df, use_container_width=True)
