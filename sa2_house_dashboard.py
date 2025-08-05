import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from io import BytesIO
import plotly.io as pio

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("SA2 Scores July 2025.xlsx", sheet_name="House", header=None)
    header_row_index = df[df.iloc[:, 1] == "SA2"].index[0]
    df.columns = df.iloc[header_row_index]
    df = df.drop(index=range(header_row_index + 1)).reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    return df

df = load_data()
st.title("PropWealth Buyers Agency")

# --- SA2 Selection for trend graphs ---
if "SA2" in df.columns:
    selected_sa2 = st.selectbox("Select SA2 to view detailed 3M & 12M trends:", sorted(df["SA2"].dropna().unique()))

    if selected_sa2:
        sa2_row = df[df["SA2"] == selected_sa2]
        if not sa2_row.empty:
            st.subheader("📊 3M and 12M Trend Comparison for Selected SA2")

            # Define metric groups to map
            metric_groups = {
                "Sale Median": ["Sale Median 12m Ago", "Sale Median 3m Ago", "Sale Median Now"],
                "List Price Median": ["List Price Median 12m Ago", "List Price Median 3m Ago", "List Price Median Now"],
                "House Median (SA3)": ["House Median 24M Ago (SA3)", "House Median 12M Ago (SA3)", "House Median Now (SA3)"],
                "Rent Median (SA3)": ["House Median Rent 12M Ago (SA3)", "House Median Rent Now (SA3)"],
                "Growth (%)": ["12m Growth (%)", "24m Growth (%)"],
                "Affordability": ["Rent Affordability (% of Income)", "Buy Affordability (Years)"],
                "Sales Turnover": ["Sales Turnover 12M Ago (%)", "Sales Turnover 3M Ago (%)", "Sales Turnover Now (%)"],
                "For Sale Listings": ["For Sale Av Listings 12m Ago (SA3)", "For Sale Av Listings 3m Ago (SA3)", "For Sale Av Listings Now (SA3)"]
            }

            for title, cols in metric_groups.items():
                values = []
                labels = []

                for col in cols:
                    if col in sa2_row.columns:
                        try:
                            val = float(sa2_row[col].values[0])
                            values.append(val)
                            labels.append(col)
                        except:
                            continue

                if values:
                    chart = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="royalblue")])
                    chart.update_layout(title=title, xaxis_title="Period", yaxis_title="Value")
                    st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("No data found for selected SA2.")
else:
    st.error("SA2 column not found in the dataset.")
