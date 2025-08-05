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
st.title("Enhanced SA2 House Dashboard")

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Advanced Filters")
    selected_filters = {}    # For dropdowns
    score_filters = {}       # For sliders

    for col in df.columns:
        col_lower = str(col).lower()

        # Handle any column with "score" using slider
        if 'score' in col_lower:
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.notnull().any():  # Ensure it's not all NaN
                    min_val = float(numeric_col.min())
                    max_val = float(numeric_col.max())
                    selected_range = st.slider(
                        f"{col}", min_value=min_val, max_value=max_val, value=(min_val, max_val)
                    )
                    score_filters[col] = (numeric_col, selected_range)
            except:
                pass  # Skip if can't convert to numeric

        # Handle categorical filters
        elif pd.api.types.is_object_dtype(df[col]):
            values = sorted(df[col].dropna().unique())
            selected = st.multiselect(f"Filter by {col}:", values)
            if selected:
                selected_filters[col] = selected

# --- Apply Filters ---
filtered_df = df.copy()

# Apply dropdown filters
for col, selected_vals in selected_filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Apply slider filters
for col, (numeric_col, (min_val, max_val)) in score_filters.items():
    # Keep same indices as df, not filtered_df
    mask = (numeric_col >= min_val) & (numeric_col <= max_val)
    filtered_df = filtered_df[mask]

# --- Show Filtered Data ---
st.dataframe(filtered_df, use_container_width=True)

# --- Download Buttons ---
st.download_button("Download Filtered Data as CSV",
                   data=filtered_df.to_csv(index=False).encode(),
                   file_name="filtered_data.csv",
                   mime="text/csv")

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="Filtered")
excel_buffer.seek(0)

st.download_button("Download Filtered Data as Excel",
                   data=excel_buffer,
                   file_name="filtered_data.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Trend Analysis ---
if "SA2" in df.columns:
    selected_sa2s = st.multiselect("Select SA2(s) to view trends:", sorted(df["SA2"].dropna().unique()))
    if selected_sa2s:
        st.subheader("Trend Comparison")
        fig = go.Figure()
        group_data = []

        for sa2 in selected_sa2s:
            sa2_data = df[df["SA2"] == sa2]
            if not sa2_data.empty:
                num_data = sa2_data.select_dtypes(include=["number", "float", "int"]).T
                num_data.columns = ["Value"]
                num_data.index.name = "Year"
                num_data.reset_index(inplace=True)
                num_data["Value"] = pd.to_numeric(num_data["Value"], errors="coerce")
                num_data = num_data.dropna(subset=["Value"])
                group_data.append((sa2, num_data["Value"].mean()))
                fig.add_trace(go.Scatter(x=num_data["Year"], y=num_data["Value"],
                                         mode="lines+markers", name=sa2))

        fig.update_layout(title="Year-wise Trends by SA2",
                          xaxis_title="Year/Metric",
                          yaxis_title="Value",
                          hovermode="x unified",
                          legend_title="SA2")
        st.plotly_chart(fig, use_container_width=True)

        # Chart export options
        for fmt in ["png", "svg", "pdf"]:
            try:
                img_data = pio.to_image(fig, format=fmt, engine="kaleido")
                mime = "image/svg+xml" if fmt == "svg" else f"image/{fmt}" if fmt == "png" else "application/pdf"
                st.download_button(f"Download Chart as {fmt.upper()}",
                                   data=img_data,
                                   file_name=f"trend_graph.{fmt}",
                                   mime=mime)
            except Exception as e:
                st.warning(f"❌ Could not export {fmt.upper()} chart: {e}")

        # Average Score Summary
        st.subheader("2020–2025 Average (Mock Grouping)")
        avg_table = pd.DataFrame(group_data, columns=["SA2", "2020–2025 Avg"])
        st.table(avg_table)

        # AI Interpretation
        st.subheader("AI Summary for Selected SA2(s)")
        for sa2, avg_val in group_data:
            if avg_val > 65:
                msg = f"🔵 {sa2} shows very strong performance based on recent trends."
            elif avg_val > 50:
                msg = f"🟡 {sa2} has moderate performance with room to grow."
            else:
                msg = f"🔴 {sa2} is currently underperforming in comparison to others."
            st.markdown(msg)
else:
    st.error("SA2 column not found in the dataset.")
