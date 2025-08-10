
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from io import BytesIO
import plotly.io as pio
import requests
from bs4 import BeautifulSoup
import json
import re

# ---------------- Basic App Setup ----------------
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
            vals = df[col].dropna().unique()
            if len(vals) and all(v in [True, False, "True", "False", "TRUE", "FALSE"] for v in vals):
                bool_like.append(col)
    for col in bool_like:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0}).astype("Int64")

    # Target columns with numeric hints
    for col in df.columns:
        name = str(col).lower()
        if any(h in name for h in NUMERIC_HINTS):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%','', regex=False).str.replace(',','', regex=False),
                errors='coerce'
            )

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

# ---------------- Utilities ----------------
def _guess_investor_score_col(columns) -> str | None:
    lower_map = {c: str(c).lower() for c in columns}
    for c, lc in lower_map.items():
        if lc.strip() in ("investor score", "investors score"):
            return c
    candidates = [c for c, lc in lower_map.items() if "investor" in lc and "score" in lc]
    if candidates:
        candidates.sort(key=lambda x: len(str(x)))
        return candidates[0]
    return None

# ---------------- Sidebar: Dynamic Filters ----------------
filter_columns = df.select_dtypes(include=['object']).columns.tolist()

with st.sidebar:
    st.header("Advanced Filters")
    selected_filters = {}
    for col in filter_columns:
        values = sorted([v for v in df[col].dropna().unique() if v != ""])
        selected = st.multiselect(f"Filter by {col}:", values)
        if selected:
            selected_filters[col] = selected

    # Investor Score slider
    investor_col = _guess_investor_score_col(df.columns)
    inv_min = inv_max = None
    if investor_col is not None:
        inv_series = pd.to_numeric(df[investor_col], errors="coerce")
        data_min = int(max(0, float(inv_series.min(skipna=True)) if pd.notna(inv_series.min(skipna=True)) else 0))
        data_max = int(float(inv_series.max(skipna=True)) if pd.notna(inv_series.max(skipna=True)) else 100)
        lower = max(0, min(100, data_min)) if data_max <= 110 else data_min
        upper = min(100, max(lower, data_max)) if data_max <= 110 else data_max

        inv_min, inv_max = st.slider(
            f"Investor Score range ({investor_col})",
            min_value=int(lower),
            max_value=int(upper),
            value=(int(lower), int(upper)),
            help="Drag to keep only rows where Investor Score falls within this range."
        )
        st.caption(f"Filtering Investor Score between **{inv_min}** and **{inv_max}**.")

# Apply filters
filtered_df = df.copy()
for col, selected_vals in selected_filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

if 'inv_min' in locals() and inv_min is not None and investor_col is not None:
    filtered_df = filtered_df[pd.to_numeric(filtered_df[investor_col], errors="coerce").between(inv_min, inv_max)]

filtered_df = _coerce_numeric_cols(filtered_df.copy())
st.dataframe(filtered_df, use_container_width=True)

# ---------------- Downloads ----------------
st.download_button(
    "Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="filtered_data.csv",
    mime="text/csv"
)

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="Filtered")
excel_buffer.seek(0)
st.download_button(
    "Download Filtered Data as Excel",
    data=excel_buffer,
    file_name="filtered_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ---------------- realestate.com.au Helpers ----------------
def build_rea_search_url(suburb:str, state_code:str, min_price:int|None=None, max_price:int|None=None, prop_type:str|None=None, beds:int|None=None, page:int=1):
    slug = f"{suburb.strip().lower().replace(' ', '-')}-{state_code.lower()}"
    base = f"https://www.realestate.com.au/buy/in-{slug}/list-{page}"
    params = []
    if min_price is not None or max_price is not None:
        lo = 0 if min_price is None else int(min_price)
        hi = "" if max_price is None else int(max_price)
        params.append(f"price={lo}-{hi}")
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
                        'address': addr.get('streetAddress') if isinstance(addr, dict) else None,
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
    seen = set()
    dedup = []
    for rrow in all_rows:
        key = rrow.get('url') or (rrow.get('title'), rrow.get('address'), rrow.get('price'))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(rrow)
    return pd.DataFrame(dedup)

# ---------------- Search UI (Restricted to Visible SA2s) ----------------
st.markdown("### Search properties on realestate.com.au and export to CSV")

def _clean_sa2_to_suburb(sa2: str) -> str:
    if not isinstance(sa2, str):
        return sa2
    base = sa2.split(" - ")[0]
    base = re.sub(r"\s*\([^)]*\)\s*", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base

if "SA2" in filtered_df.columns and not filtered_df.empty:
    visible_sa2 = (
        filtered_df["SA2"].dropna().astype(str).str.strip().unique().tolist()
    )
    if not visible_sa2:
        st.info("No suburbs visible after filters. Adjust filters to enable search.")
    else:
        sa2_to_suburb = {sa2: _clean_sa2_to_suburb(sa2) for sa2 in visible_sa2}

        state_col = "State" if "State" in filtered_df.columns else None
        if state_col:
            visible_states = (
                filtered_df.loc[filtered_df["SA2"].isin(visible_sa2), state_col]
                .dropna().astype(str).str.upper().str.strip().unique().tolist()
            )
        else:
            visible_states = []

        suburb_for_search_sa2 = st.selectbox(
            "Choose a suburb (from current dashboard):",
            sorted(visible_sa2),
            help="Only suburbs visible after applying the sidebar filters are listed."
        )
        suburb_clean = sa2_to_suburb.get(suburb_for_search_sa2, suburb_for_search_sa2)

        if len(visible_states) == 1:
            state_code = visible_states[0]
            st.caption(f"State auto-detected from filtered data: **{state_code}**")
        else:
            state_options = visible_states if visible_states else ["VIC","NSW","QLD","WA","SA","ACT","TAS","NT"]
            default_idx = state_options.index("NSW") if "NSW" in state_options else 0
            state_code = st.selectbox("State code:", state_options, index=default_idx)

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

        base_url = build_rea_search_url(
            suburb_clean,
            state_code,
            min_price if min_price > 0 else None,
            max_price if max_price > 0 else None,
            prop_type,
            beds
        )
        st.code(base_url, language="text")
        st.link_button(f"Open REA search for {suburb_clean}, {state_code}", base_url)

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
                st.download_button(
                    "Download listings.csv",
                    data=csv_bytes,
                    file_name=f"{suburb_clean.replace(' ','_').lower()}_listings.csv",
                    mime="text/csv"
                )
else:
    st.info("Apply filters or load data so SA2 (suburb) options appear on the dashboard first.")

# ---------------- Suburb Trends: Price & Days on Market ----------------
st.subheader("Suburb Trends: Price & Days on Market")

def _find_timeseries_columns(columns, metric_keywords):
    """
    Return a list of columns that look like time-series for the metric:
    - Column name must contain one of metric_keywords
    - And contain a year-like token (2019..2030) or month name/abbrev
    """
    months = ("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
              "january","february","march","april","june","july","august","september","october","november","december")
    out = []
    for c in columns:
        lc = str(c).lower()
        if any(k in lc for k in metric_keywords) and (re.search(r'(19|20)\d{2}', lc) or any(m in lc for m in months)):
            out.append(c)
    return out

def _extract_period(name: str):
    """Try to map a column name to an ordered period string for plotting/sorting."""
    n = str(name)
    # Prefer YYYY-MM in name
    y = re.search(r'((19|20)\d{2})', n)
    m = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', n.lower())
    if y and m:
        return f"{y.group(1)}-{m.group(1).title()}"
    if y:
        return y.group(1)
    if m:
        return m.group(1).title()
    return n

def _plot_metric_trend(metric_name, metric_keywords):
    cols_ts = _find_timeseries_columns(df.columns, metric_keywords)
    if cols_ts:
        # Build tidy df: period, SA2, value
        periods = {c: _extract_period(c) for c in cols_ts}
        # Keep periods sorted by (year, month) if possible
        def _period_key(p):
            y = re.search(r'(19|20)\d{2}', p)
            m_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
            m = None
            for k,v in m_map.items():
                if k.lower() in p.lower():
                    m = v; break
            return (int(y.group(0)) if y else 0, m if m else 0, p)

        ordered = sorted(periods.items(), key=lambda kv: _period_key(kv[1]))
        ordered_cols = [c for c,_ in ordered]

        selected_sa2s = st.multiselect(f"Select SA2(s) for {metric_name} trend:", sorted(filtered_df["SA2"].dropna().unique()), key=f"{metric_name}_ts_sa2")
        if not selected_sa2s:
            st.info(f"Pick one or more SA2s above to see {metric_name} trends.")
            return

        fig = go.Figure()
        for sa2 in selected_sa2s:
            sub = df[df["SA2"] == sa2]
            if sub.empty:
                continue
            yvals = []
            xvals = []
            for c in ordered_cols:
                val = pd.to_numeric(sub[c], errors="coerce")
                if not val.empty and pd.notna(val.iloc[0]):
                    yvals.append(float(val.iloc[0]))
                    xvals.append(periods[c])
            if xvals and yvals:
                fig.add_trace(go.Scatter(x=xvals, y=yvals, mode="lines+markers", name=sa2))
        fig.update_layout(title=f"{metric_name} Trend", xaxis_title="Period", yaxis_title=metric_name, hovermode="x unified", legend_title="SA2")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: plot current levels (bar) if we only have a single column
        # Try to find a single current metric column
        single_candidates = [c for c in df.columns if any(k in str(c).lower() for k in metric_keywords)]
        single_col = None
        if single_candidates:
            # Prefer ones with 'median' for price and 'days'/'dom' for DOM, else pick shortest
            if any('median' in str(c).lower() for c in single_candidates):
                single_col = [c for c in single_candidates if 'median' in str(c).lower()][0]
            else:
                single_candidates.sort(key=lambda x: len(str(x)))
                single_col = single_candidates[0]

        if single_col is None:
            st.warning(f"No columns found for {metric_name}.")
            return

        # Show current levels for visible SA2s
        sub = filtered_df[["SA2", single_col]].dropna()
        if sub.empty:
            st.info(f"No data to plot for {metric_name} in the current selection.")
            return
        sub = sub.sort_values(by=single_col, ascending=False).head(25)  # top 25 for readability
        fig = go.Figure(data=[go.Bar(x=sub["SA2"].astype(str), y=pd.to_numeric(sub[single_col], errors="coerce"))])
        fig.update_layout(title=f"{metric_name} (Current Levels)", xaxis_title="SA2", yaxis_title=single_col, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Price trend
_plot_metric_trend("Median Price", metric_keywords=("price","median price","median_value","median"))
# Days on Market trend
_plot_metric_trend("Days on Market", metric_keywords=("days on market","dom","days"))

# ---------------- Trend Analysis (Generic) ----------------
if "SA2" in df.columns:
    selected_sa2s = st.multiselect("Select SA2(s) to view generic trends:", sorted(df["SA2"].dropna().unique()), key="generic_trends_sa2")
    if selected_sa2s:
        st.subheader("Generic Trend Comparison")
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

        for fmt in ["png", "svg", "pdf"]:
            try:
                img_data = pio.to_image(fig, format=fmt, engine="kaleido")
                mime = "image/svg+xml" if fmt == "svg" else f"image/{fmt}" if fmt == "png" else "application/pdf"
                st.download_button(f"Download Generic Trend as {fmt.upper()}", data=img_data, file_name=f"trend_graph.{fmt}", mime=mime)
            except Exception as e:
                st.warning(f"❌ Could not export {fmt.upper()} chart: {e}")
else:
    st.error("SA2 column not found in the dataset.")
