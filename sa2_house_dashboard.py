
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "SA2 Scores July 2025.xlsx"
    df = pd.read_excel(file_path, sheet_name="House", header=None)

    # Identify the header row dynamically where 'SA2' appears in the second column
    header_row_index = df[df.iloc[:, 1] == "SA2"].index[0]
    df.columns = df.iloc[header_row_index]
    df = df.drop(index=range(header_row_index + 1)).reset_index(drop=True)

    # Drop empty columns
    df = df.dropna(axis=1, how='all')
    return df

df = load_data()

st.title("SA2 House Dashboard")

required_cols = ["State", "Property\nType", "SA2"]
if all(col in df.columns for col in required_cols):
    with st.sidebar:
        st.header("Filters")
        selected_states = st.multiselect("Select State(s):", sorted(df["State"].dropna().unique()))
        selected_types = st.multiselect("Select Property Type(s):", sorted(df["Property\nType"].dropna().unique()))
        selected_sa2 = st.selectbox("Select SA2 for Trend Graph:", sorted(df["SA2"].dropna().unique()))

    # Apply Filters
    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]
    if selected_types:
        filtered_df = filtered_df[filtered_df["Property\nType"].isin(selected_types)]

    st.dataframe(filtered_df, use_container_width=True)

    # Trend Graph
    sa2_data = df[df["SA2"] == selected_sa2]
    numeric_data = sa2_data.select_dtypes(include=["number", "float", "int"]).T
    numeric_data.columns = ["Value"]
    numeric_data.index.name = "Year"
    numeric_data.reset_index(inplace=True)

    # Ensure the x and y data are numeric-compatible
    try:
        numeric_data["Value"] = pd.to_numeric(numeric_data["Value"], errors="coerce")
        numeric_data = numeric_data.dropna(subset=["Value"])

        if not numeric_data.empty:
            st.subheader(f"Trend Graph for SA2: {selected_sa2}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(numeric_data["Year"], numeric_data["Value"], marker="o", label=selected_sa2)
            ax.set_xlabel("Year/Metric")
            ax.set_ylabel("Value")
            ax.set_title(f"Year-wise Trends for {selected_sa2}")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Export to PDF
            buffer = BytesIO()
            fig.savefig(buffer, format="pdf")
            buffer.seek(0)
            st.download_button("Download Trend Graph as PDF", data=buffer, file_name=f"{selected_sa2}_trend.pdf", mime="application/pdf")
        else:
            st.warning("No valid numeric trend data available to plot for the selected SA2.")
    except Exception as e:
        st.error(f"An error occurred while generating the graph: {e}")
else:
    missing = [col for col in required_cols if col not in df.columns]
    st.error("Missing required column(s): " + ", ".join(missing))
