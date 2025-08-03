# Overwrite the Streamlit script in the zip bundle with the final fixed version
final_fixed_script_path = "/mnt/data/sa2_house_dashboard.py"

# Rewriting the script content directly
final_dashboard_code = """
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "SA2 Scores July 2025.xlsx"
    df = pd.read_excel(file_path, sheet_name="House", header=None)

    # Dynamically locate the header row and reset the DataFrame
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
    type_col = "Property\\nType"

    if state_col in df.columns and type_col in df.columns:
        states = st.multiselect("Select State(s):", sorted(df[state_col].dropna().unique()))
        property_types = st.multiselect("Select Property Type(s):", sorted(df[type_col].dropna().unique()))

        # Apply Filters
        filtered_df = df.copy()
        if states:
            filtered_df = filtered_df[filtered_df[state_col].isin(states)]
        if property_types:
            filtered_df = filtered_df[filtered_df[type_col].isin(property_types)]

        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.error("The dataset is missing required columns.")
"""

# Save it to replace the broken version in the bundle
with open(final_fixed_script_path, "w") as f:
    f.write(final_dashboard_code)

# Rebuild the zip bundle with corrected contents
fixed_zip_path = "/mnt/data/SA2_House_Dashboard_Bundle_Fixed.zip"
with zipfile.ZipFile(fixed_zip_path, "w") as zipf:
    zipf.write(final_fixed_script_path, arcname="sa2_house_dashboard.py")
    zipf.write("/mnt/data/SA2 Scores July 2025.xlsx", arcname="SA2 Scores July 2025.xlsx")
    zipf.write("/mnt/data/requirements.txt", arcname="requirements.txt")

fixed_zip_path
