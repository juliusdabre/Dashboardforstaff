
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "SA2 Scores July 2025.xlsx"
    df = pd.read_excel(file_path, sheet_name="House", header=None)
    df.columns = df.iloc[1]
    df = df.drop([0, 1]).reset_index(drop=True)
    return df

df = load_data()

st.title("SA2 House Dashboard")

# Sidebar Filters
with st.sidebar:
    st.header("Filters")
    states = st.multiselect("Select State(s):", sorted(df["State"].dropna().unique()))
    property_types = st.multiselect("Select Property Type(s):", sorted(df["Property\nType"].dropna().unique()))

# Apply Filters
filtered_df = df.copy()
if states:
    filtered_df = filtered_df[filtered_df["State"].isin(states)]
if property_types:
    filtered_df = filtered_df[filtered_df["Property\nType"].isin(property_types)]

# Display Data
st.dataframe(filtered_df, use_container_width=True)
