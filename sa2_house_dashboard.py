
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from io import BytesIO
import plotly.io as pio
import requests
from bs4 import BeautifulSoup
import json
import re

st.set_page_config(page_title="SA2 House Dashboard", layout="wide")

NUMERIC_HINTS = (
    "score", "yield", "growth", "inventory", "turnover", "days", "dom", "vacancy",
    "rent", "price", "afford", "median", "months", "percent", "index"
)

def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Convert obvious boolean-like columns to ints first
    bool_like = []
    for col in df.columns:
        if df[col].dtype == "bool":
            bool_like.append(col)
        else:
            # objects that look like booleans mixed in
            vals = df[col].dropna().unique()
            if len(vals) and all(v in [True, False, "True", "False", "TRUE", "FALSE"] for v in vals):
                bool_like.append(col)
    for col in bool_like:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0}).astype("Int64")

    # Target columns with numeric hints in the name for coercion
    for col in df.columns:
        name = str(col).lower()
        if any(h in name for h in NUMERIC_HINTS):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%','', regex=False).str.replace(',','', regex=False), errors='coerce')

    # Specific fix for known column in logs
    if "Sales Turnover Score (SA2)" in df.columns:
        df["Sales Turnover Score (SA2)"] = pd.to_numeric(df["Sales Turnover Score (SA2)"], errors="coerce")

    return df

@st.cache_data
def load_data():
    df = pd.read_excel("SA2 Scores July 2025.xlsx", sheet_name="House", header=None)
    header_row_index = df[df.iloc[:, 1] == "SA2"].index[0]
    df.columns = df.iloc[header_row_index]
    df = df.drop(index=range(header_row_index + 1)).reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    df = _coerce_numeric_cols(df.copy())
    return df

df = load_data()
st.title("PropWealth Buyers Agency")

# Dynamically list all possible filters based on non-numeric columns
filter_columns = df.select_dtypes(include=['object']).columns.tolist()

with st.sidebar:
    st.header("Advanced Filters")
    selected_filters = {}
    for col in filter_columns:
        values = sorted([v for v in df[col].dropna().unique() if v != ""])
        selected = st.multiselect(f"Filter by {col}:", values)
        if selected:
            selected_filters[col] = selected

filtered_df = df.copy()
for col, selected_vals in selected_filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Ensure Arrow compatibility before rendering
filtered_df = _coerce_numeric_cols(filtered_df.copy())
st.dataframe(filtered_df, use_container_width=True)

# CSV and Excel download
st.download_button("Download Filtered Data as CSV", data=filtered_df.to_csv(index=False).encode(), file_name="filtered_data.csv", mime="text/csv")

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="Filtered")
excel_buffer.seek(0)
st.download_button("Download Filtered Data as Excel", data=excel_buffer, file_name="filtered_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- realestate.com.au SEARCH + SCRAPE TO CSV ----------
def build_rea_search_url(suburb:str, state_code:str, min_price:int|None=None, max_price:int|None=None, prop_type:str|None=None, beds:int|None=None, page:int=1):
    # suburb slug like 'cranbourne-vic'
    slug = f"{suburb.strip().lower().replace(' ', '-')}-{state_code.lower()}"
    base = f"https://www.realestate.com.au/buy/in-{slug}/list-{page}"
    params = []
    if min_price is not None or max_price is not None:
        lo = 0 if min_price is None else int(min_price)
        hi = "" if max_price is None else int(max_price)
        params.append(f"price={lo}-{hi}")
    # property type mapping
    pt_map = {
        "Any": "",
        "House": "property-house",
        "Townhouse": "property-townhouse",
        "Apartment/Unit": "property-unit+apartment",
        "Land": "property-land",
        "Villa": "property-villa",
        "Duplex": "property-duplex"
    }
    if prop_type and pt_map.get(prop_type):
        base = base.replace("/buy", f"/buy/{pt_map[prop_type]}")
    if beds:
        params.append(f"bedrooms={beds}-any")
    if params:
        return base + "?" + "&".join(params)
    return base

def _first_text(el, selectors):
    for sel in selectors:
        hit = el.select_one(sel)
        if hit:
            t = hit.get_text(" ", strip=True)
            if t:
                return t
    return None

def _first_attr(el, selectors, attr):
    for sel in selectors:
        hit = el.select_one(sel)
        if hit and hit.has_attr(attr):
            return hit[attr]
    return None

def parse_cards_html(soup):
    rows = []
    cards = soup.select('[data-testid="listing-card"], article, li[data-testid*="search-result"]')
    for c in cards:
        title = _first_text(c, ['[data-testid="listing-card-title"]','h2','h3','a[aria-label]'])
        address = _first_text(c, ['[data-testid="address-label"]','[data-testid="listing-card-subtitle"]','.property-info__address','[itemprop="streetAddress"]','[data-testid="property-card-subtitle"]'])
        price = _first_text(c, ['[data-testid="listing-card-price"]','.property-price','[aria-label*="price"]','[data-testid="property-price"]'])
        link = _first_attr(c, ['a[href*="/property-"]','a[data-testid="listing-card-link"]','a[aria-label]'], 'href')
        beds = _first_text(c, ['[data-testid="property-features-text-container"]','[data-testid="listing-card-bedroom"]','.general-features__beds','.property-feature__beds'])
        baths = _first_text(c, ['[data-testid="listing-card-bathroom"]','.general-features__baths','.property-feature__baths'])
        car = _first_text(c, ['[data-testid="listing-card-carspace"]','.general-features__cars','.property-feature__cars'])
        if link and link.startswith('/'):
            link = 'https://www.realestate.com.au' + link
        # Normalize features
        def parse_num(txt):
            if not txt:
                return None
            m = re.search(r'(\d+(?:\.\d+)?)', txt)
            return float(m.group(1)) if m else None
        rows.append({
            'title': title,
            'address': address,
            'price': price,
            'beds': parse_num(beds),
            'baths': parse_num(baths),
            'car': parse_num(car),
            'url': link
        })
    return [r for r in rows if any(v for v in r.values())]

def parse_ld_json(soup):
    rows = []
    for s in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(s.string)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        for obj in data:
            if isinstance(obj, dict) and (obj.get('@type') in ('Offer','Residence','SingleFamilyResidence','Apartment','House') or ('@graph' in obj)):
                items = obj.get('@graph', [obj]) if '@graph' in obj else [obj]
                for it in items:
                    addr = it.get('address', {})
                    price = None
                    if 'offers' in it and isinstance(it['offers'], dict):
                        price = it['offers'].get('price') or it['offers'].get('lowPrice')
                    rows.append({
                        'title': it.get('name'),
                        'address': addr.get('streetAddress'),
                        'price': price,
                        'beds': it.get('numberOfRooms'),
                        'baths': it.get('numberOfBathroomsTotal') or it.get('numberOfBathrooms'),
                        'car': None,
                        'url': it.get('url')
                    })
    return [r for r in rows if any(v for v in r.values())]

def fetch_rea_listings(search_url, pages=1, timeout=15):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-AU,en;q=0.9",
    }
    all_rows = []
    for p in range(1, pages+1):
        url = re.sub(r'/list-\d+', f'/list-{p}', search_url)
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
        except Exception as e:
            st.warning(f"Failed to fetch page {p}: {e}")
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = parse_cards_html(soup)
        if not rows:
            rows = parse_ld_json(soup)
        all_rows.extend(rows)
    # Deduplicate by URL
    seen = set()
    dedup = []
    for r in all_rows:
        key = r.get('url') or (r.get('title'), r.get('address'), r.get('price'))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return pd.DataFrame(dedup)

st.markdown("### Search properties on realestate.com.au and export to CSV")
if "SA2" in filtered_df.columns:
    state_code = st.selectbox("State code (e.g., VIC/NSW/WA/SA/QLD/ACT/TAS/NT):", ["VIC","NSW","QLD","WA","SA","ACT","TAS","NT"], index=1)
    suburb_for_search = st.selectbox("Choose a suburb to search:", sorted(filtered_df["SA2"].dropna().unique()))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_price = st.number_input("Min Price ($)", min_value=0, step=25000, value=0)
    with col2:
        max_price = st.number_input("Max Price ($)", min_value=0, step=25000, value=0)
    with col3:
        prop_type = st.selectbox("Property Type", ["Any","House","Townhouse","Apartment/Unit","Land","Villa","Duplex"])
    with col4:
        beds = st.selectbox("Min Beds", [None,1,2,3,4,5], index=0)
    max_pages = st.slider("Pages to fetch", 1, 10, 3)
    base_url = build_rea_search_url(suburb_for_search, state_code, min_price if min_price>0 else None, max_price if max_price>0 else None, prop_type, beds)
    st.code(base_url, language="text")
    st.link_button(f"Open REA search for {suburb_for_search}", base_url)
    custom_url = st.text_input("Or paste a custom realestate.com.au search URL:", value=base_url)
    if st.button("Fetch listings & make CSV"):
        with st.spinner("Fetching listings..."):
            df_list = fetch_rea_listings(custom_url, pages=max_pages)
        if df_list.empty:
            st.error("No listings parsed. Try increasing pages or paste the exact search URL from your browser.")
        else:
            st.success(f"Fetched {len(df_list)} listings")
            st.dataframe(df_list, use_container_width=True)
            csv_bytes = df_list.to_csv(index=False).encode()
            st.download_button("Download listings.csv", data=csv_bytes, file_name=f"{suburb_for_search.replace(' ','_').lower()}_listings.csv", mime="text/csv")
else:
    st.info("Filter or load data so SA2 (suburb) options appear.")

# ---------- Trend analysis ----------
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
                fig.add_trace(go.Scatter(x=num_data["Year"], y=num_data["Value"], mode="lines+markers", name=sa2))

        fig.update_layout(title="Year-wise Trends by SA2", xaxis_title="Year/Metric", yaxis_title="Value", hovermode="x unified", legend_title="SA2")
        st.plotly_chart(fig, use_container_width=True)

        # Export chart safely
        for fmt in ["png", "svg", "pdf"]:
            try:
                img_data = pio.to_image(fig, format=fmt, engine="kaleido")
                mime = "image/svg+xml" if fmt == "svg" else f"image/{fmt}" if fmt == "png" else "application/pdf"
                st.download_button(f"Download Chart as {fmt.upper()}", data=img_data, file_name=f"trend_graph.{fmt}", mime=mime)
            except Exception as e:
                st.warning(f"❌ Could not export {fmt.upper()} chart: {e}")

        # Grouped Summary
        st.subheader("2020–2025 Average (Mock Grouping)")
        avg_table = pd.DataFrame(group_data, columns=["SA2", "2020–2025 Avg"])
        st.table(avg_table)

        # AI Summary
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
