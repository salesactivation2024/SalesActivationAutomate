import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from io import BytesIO
import streamlit as st
import pandas as pd
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from io import BytesIO
import re
from datetime import timedelta
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
import streamlit as st
import urllib


st.set_page_config(page_title="Outlet Management", layout="wide")
st.title("🏪 Outlet Management Dashboard")

@st.cache_resource
def make_engine():
    s = st.secrets["dbo"]
    odbc_str = (
        f"DRIVER={{{s['driver']}}};"
        f"SERVER={s['host']},{s['port']};DATABASE={s['database']};UID={s['username']};PWD={s['password']};"
        "Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
    )
    url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str)

    engine = create_engine(url)
    return engine

engine = make_engine()


# ============================ Table names ============================
T_SMSDV_DESC = st.secrets.get("tables", {}).get("smsdv_desc", "dbo.smsdv_desc")
T_SMSDV      = st.secrets.get("tables", {}).get("smsdv", "dbo.smsdv")
T_MNT        = st.secrets.get("tables", {}).get("mnt", "dbo.mnt")
T_DAKAR      = st.secrets.get("tables", {}).get("dakar", "dbo.dakar")

# ============================ Helpers ============================
@st.cache_data(ttl=900, show_spinner=False)
def get_columns(table: str) -> list[str]:
    try:
        q = text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = PARSENAME(:t,1)
              AND (TABLE_SCHEMA = PARSENAME(:t,2) OR :t NOT LIKE '%.%')
            ORDER BY ORDINAL_POSITION
        """)
        return pd.read_sql(q, engine, params={"t": table})["COLUMN_NAME"].str.strip().tolist()
    except Exception:
        return []

def pick(available: list[str], *cands: str) -> str | None:
    if not available:
        return None
    low = {a.lower(): a for a in available}
    for c in cands:
        if c and c.lower() in low:
            return low[c.lower()]
    for c in cands:
        if not c:
            continue
        want = c.lower().replace(" ", "").replace("_", "")
        for a in available:
            if a.lower().replace(" ", "").replace("_", "") == want:
                return a
    return None

def norm8(alias: str, col: str) -> str:
    return f"RIGHT(REPLICATE('0',8) + CAST(TRY_CONVERT(BIGINT, {alias}.{col}) AS VARCHAR(32)), 8)"

# ============================ Detect schemas ============================
ALL_SALE_DESC = get_columns(T_SMSDV_DESC)
ALL_SALE      = get_columns(T_SMSDV)
ALL_MNT       = get_columns(T_MNT)
ALL_DKR       = get_columns(T_DAKAR)
USE_DAKAR     = len(ALL_DKR) > 0

# Choose sales table (prefer DESC)
def choose_sales_table():
    date_desc = pick(ALL_SALE_DESC, "converted_date", "Tanggal", "Date")
    cust_desc = pick(ALL_SALE_DESC, "Customer_Code", "Customer", "Cust_Code",
                     "Outlet_Code", "Outlet_ID", "Customer_ID", "Customer No", "Outlet")
    if ALL_SALE_DESC and date_desc and cust_desc:
        return T_SMSDV_DESC, ALL_SALE_DESC
    return T_SMSDV, ALL_SALE

T_SALE, ALL_SALE_ACTIVE = choose_sales_table()

# ---- MNT columns (also used as fallback org)
CUST_O = pick(ALL_MNT, "Customer", "Customer_Code", "Cust_Code", "Outlet_Code", "Outlet_ID", "Customer_ID", "Customer No", "Outlet")
REG_O  = pick(ALL_MNT, "Region", "Region_Code", "Area")
CITY_O = pick(ALL_MNT, "City", "MD_Region", "City_1", "Kota")
DISTRICT_O = pick(ALL_MNT, "MD_Area", "District", "Kecamatan", "Area")
WH_O   = pick(ALL_MNT, "Ws_Name", "WH_Name", "Warehouse", "From_W_H", "WH", "Depo")
DIV_O  = pick(ALL_MNT, "Division", "Divisi")      
AMO_O  = pick(ALL_MNT, "AMO", "Sales_Office")      
CH_O   = pick(ALL_MNT, "Channel")   
SR_O   = pick(ALL_MNT, "Employee_Name", "Sales_Man_Name", "Salesman", "Sales_Rep", "SR_Name")  

DIV_D = pick(ALL_DKR, "Division", "Divisi")
AMO_D = pick(ALL_DKR, "AMO", "Sales_Office", "Sales_Office_Name", "Sales Office")
WH_D  = pick(ALL_DKR, "Ws_Name", "WH_Name", "Warehouse", "From_W_H", "WH", "Depo")
CH_D  = pick(ALL_DKR, "Channel", "Channel_Low", "Channel_Low_", "Segmen", "Segment")
REG_D = pick(ALL_DKR, "Region", "Region_Code", "Area")
SR_D  = pick(ALL_DKR, "Employee_Name", "Sales_Man_Name", "Salesman", "Sales_Rep", "SR_Name", "Nama_SR", "Nama SR")

DATE_S = pick(ALL_SALE_ACTIVE, "converted_date", "Tanggal", "Date")
CUST_S = pick(ALL_SALE_ACTIVE, "Customer_Code", "Customer", "Cust_Code",
              "Outlet_Code", "Outlet_ID", "Customer_ID", "Customer No", "Outlet")
SKU_S  = pick(ALL_SALE_ACTIVE, "SKU", "SKU_Name", "Prod_Name", "Product")

if not (CUST_O and CUST_S and DATE_S):
    st.error("Cannot detect Customer/Date columns. Please check your database schema.")
    st.stop()

DIV_SRC = ('d', DIV_D) if USE_DAKAR and DIV_D else ('o', DIV_O)
AMO_SRC = ('d', AMO_D) if USE_DAKAR and AMO_D else ('o', AMO_O)
REG_SRC = ('d', REG_D) if USE_DAKAR and REG_D else ('o', REG_O)
CH_SRC  = ('d', CH_D)  if USE_DAKAR and CH_D  else ('o', CH_O)
SR_SRC  = ('d', SR_D)  

WH_SRC  = ('o', WH_O) if WH_O else (('d', WH_D) if USE_DAKAR and WH_D else (None, None))

# ============================ Date bounds & picker ============================
@st.cache_data(ttl=900, show_spinner=False)
def get_date_bounds():
    try:
        r = pd.read_sql(text(f"SELECT CAST(MIN({DATE_S}) AS date) mn, CAST(MAX({DATE_S}) AS date) mx FROM {T_SALE}"), engine)
        if r.empty or pd.isna(r.loc[0, "mn"]) or pd.isna(r.loc[0, "mx"]):
            today = dt.date.today()
            return today, today
        return r.loc[0, "mn"], r.loc[0, "mx"]
    except Exception:
        today = dt.date.today()
        return today, today

# Get date bounds
dmin, dmax = get_date_bounds()

# ============================ Distinct list helper ============================
@st.cache_data(ttl=900, show_spinner=False)
def distinct_values(alias: str, col: str, sel):
    if not col:
        return []
    conds, params = [], {}

    def add(a, c, key, values):
        if not c or not values:
            return
        toks = []
        for i, v in enumerate(values):
            k = f"{key}{i}"
            params[k] = v
            toks.append(f":{k}")
        conds.append(f"{a}.{c} IN ({','.join(toks)})")

    # cascading filters (only include those that exist)
    if DIV_SRC[1]: add(DIV_SRC[0], DIV_SRC[1], 'div', sel.get('div'))
    if AMO_SRC[1]: add(AMO_SRC[0], AMO_SRC[1], 'amo', sel.get('amo'))
    if WH_SRC[1]:  add(WH_SRC[0],  WH_SRC[1],  'wh',  sel.get('wh'))
    if CH_SRC[1]:  add(CH_SRC[0],  CH_SRC[1],  'ch',  sel.get('ch'))
    if REG_SRC[1]: add(REG_SRC[0], REG_SRC[1], 'rg',  sel.get('rg'))
    if SR_SRC[1]:  add(SR_SRC[0],  SR_SRC[1],  'sr',  sel.get('sr'))

    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    from_sql = f"FROM {T_MNT} o " + (f"LEFT JOIN {T_DAKAR} d ON o.{REG_O} = d.{REG_D} " if USE_DAKAR and REG_O and REG_D else "")
    q = text(f"SELECT DISTINCT {alias}.{col} AS v {from_sql} {where_sql} ORDER BY v")
    
    try:
        return pd.read_sql(q, engine, params=params)["v"].astype(str).tolist()
    except Exception:
        return []

# ============================ Sidebar Configuration ============================
with st.sidebar:
    st.subheader("Display Configuration")
    
    # Frequency and SKU caps configuration with more options
    st.write("**Bucket Configuration**")
    freq_options = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    sku_options = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    
    freq_cap = st.selectbox("Purchase Frequency Cap", freq_options, index=1, format_func=lambda x: f">{x}")
    sku_cap = st.selectbox("SKU Count Cap", sku_options, index=2, format_func=lambda x: f">{x}")
    
    st.write("---")
    st.write("**Summary Options**")
    show_division_summary = st.checkbox("Show Division Summary", value=True)
    show_amo_summary = st.checkbox("Show AMO Summary", value=True)  
    show_depo_summary = st.checkbox("Show Depo/Warehouse Summary", value=True)

# ============================ Main Filters UI (reactive) ============================
st.subheader("Filters")

# Date filter at the top
dr = st.date_input("📅 Sales Date Range (affects frequency & SKU analysis)", (dmin, dmax), min_value=dmin, max_value=dmax)

# Check if dr is a tuple/list with 2 values, otherwise convert it
if isinstance(dr, (tuple, list)) and len(dr) == 2:
    d1, d2 = dr
else:
    d1, d2 = dr, dr

if d1 > d2:
    d1, d2 = d2, d1

sel = {}

c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1.6])
with c1:
    div_choices = distinct_values(DIV_SRC[0], DIV_SRC[1], sel) if DIV_SRC[1] else []
    div_sel = st.multiselect("Division", div_choices, default=[])
    sel['div'] = div_sel

with c2:
    amo_choices = distinct_values(AMO_SRC[0], AMO_SRC[1], sel) if AMO_SRC[1] else []
    amo_sel = st.multiselect("AMO / Sales Office", amo_choices, default=[])
    sel['amo'] = amo_sel

with c3:
    wh_choices = distinct_values(WH_SRC[0], WH_SRC[1], sel) if WH_SRC[1] else []
    wh_sel = st.multiselect("Warehouse (WH / Depo)", wh_choices, default=[])
    sel['wh'] = wh_sel

with c4:
    ch_choices = distinct_values(CH_SRC[0], CH_SRC[1], sel) if CH_SRC[1] else []
    ch_sel = st.multiselect("Channel", ch_choices, default=[])
    sel['ch'] = ch_sel

c5, c6 = st.columns(2)
with c5:
    reg_choices = distinct_values(REG_SRC[0], REG_SRC[1], sel) if REG_SRC[1] else []
    reg_sel = st.multiselect("Region", reg_choices, default=[])
    sel['rg'] = reg_sel

with c6:
    sr_choices = distinct_values(SR_SRC[0], SR_SRC[1], sel) if SR_SRC[1] else []
    sr_sel = st.multiselect("Sales Representative", sr_choices, default=[])
    sel['sr'] = sr_sel

# ============================ WHERE builders ============================
def build_where_for_org(sel):
    conds, params = [], {}
    def add(a, c, key):
        vals = sel.get(key)
        if c and vals:
            toks = []
            for i, v in enumerate(vals):
                k = f"{key}{i}"
                params[k] = v
                toks.append(f":{k}")
            conds.append(f"{a}.{c} IN ({','.join(toks)})")

    if DIV_SRC[1]: add(DIV_SRC[0], DIV_SRC[1], 'div')
    if AMO_SRC[1]: add(AMO_SRC[0], AMO_SRC[1], 'amo')
    if WH_SRC[1]:  add(WH_SRC[0],  WH_SRC[1],  'wh')
    if CH_SRC[1]:  add(CH_SRC[0],  CH_SRC[1],  'ch')
    if REG_SRC[1]: add(REG_SRC[0], REG_SRC[1], 'rg')
    if SR_SRC[1]:  add(SR_SRC[0],  SR_SRC[1],  'sr')

    return (" WHERE " + " AND ".join(conds)) if conds else "", params

where_org_sql, org_params = build_where_for_org(sel)

# Build sales where conditions
sale_conds = [f"CAST(s.{DATE_S} AS date) BETWEEN :d1 AND :d2"]
sale_params = {"d1": d1, "d2": d2}
if where_org_sql:
    sale_params.update(org_params)
    sale_conds.append(where_org_sql.replace(" WHERE ", ""))
where_sale = " WHERE " + " AND ".join(sale_conds)

# ============================ Outlet Master ============================
okey = norm8('o', CUST_O)

def col_expr(src, alias):
    a, c = src
    if not c:
        return "NULL AS " + alias
    return f"{a}.{c} AS {alias}"

# Join MNT with DAKAR on Region for better data linking
from_master = f"FROM {T_MNT} o " + (f"LEFT JOIN {T_DAKAR} d ON o.{REG_O} = d.{REG_D} " if USE_DAKAR and REG_O and REG_D else "")

select_m = f"""
    SELECT DISTINCT {okey} AS outlet,
           o.{CUST_O} AS Customer_Code,
           {col_expr(DIV_SRC, 'Division')},
           {col_expr(AMO_SRC, 'AMO')},
           {col_expr(REG_SRC, 'Region')},
           o.{REG_O} AS Region_MNT,
           {('o.'+CITY_O+' AS City,') if CITY_O else 'NULL AS City,'}
           {('o.'+DISTRICT_O+' AS District,') if DISTRICT_O else 'NULL AS District,'}
           {col_expr(CH_SRC, 'Channel')},
           {col_expr(WH_SRC, 'WH')},
           {col_expr(SR_SRC, 'SR')}
    {from_master}
    {where_org_sql}
"""

with st.spinner("Loading outlet master data..."):
    try:
        m = pd.read_sql(text(select_m), engine, params=org_params)
    except Exception as e:
        st.error(f"Error loading outlet master: {str(e)}")
        st.stop()

# ============================ Sales Aggregations ============================
skey = norm8('s', CUST_S)
from_sales = f"FROM {T_SALE} s INNER JOIN {T_MNT} o ON {skey} = {okey} " + (f"LEFT JOIN {T_DAKAR} d ON o.{REG_O} = d.{REG_D} " if USE_DAKAR and REG_O and REG_D else "")

# Purchase Frequency: count distinct dates per customer (filtered by date range and org)
sql_freq = text(f"""
    SELECT {okey} AS outlet, COUNT(DISTINCT CAST(s.{DATE_S} AS date)) AS freq
    {from_sales}
    {where_sale}
    GROUP BY {okey}
""")

# SKU Count: distinct SKU per customer (filtered by date range and org)
sql_sku = text(f"""
    SELECT {okey} AS outlet, COUNT(DISTINCT s.{SKU_S}) AS sku_n
    {from_sales}
    {where_sale}
    GROUP BY {okey}
""") if SKU_S else None

with st.spinner("Loading sales summaries..."):
    try:
        freq = pd.read_sql(sql_freq, engine, params=sale_params)
        sku = pd.read_sql(sql_sku, engine, params=sale_params) if sql_sku is not None else pd.DataFrame(columns=["outlet","sku_n"])
    except Exception as e:
        st.error(f"Error loading sales data: {str(e)}")
        st.stop()

# ============================ Combine & Create Buckets ============================
out = m.copy()

# Ensure outlet column is consistent format
for _df in (out, freq, sku):
    if "outlet" in _df.columns:
        _df["outlet"] = _df["outlet"].astype(str).str.strip()

# Merge frequency and SKU data
out = out.merge(freq, on="outlet", how="left").merge(sku, on="outlet", how="left")
out["freq"] = out["freq"].fillna(0).astype(int)
out["sku_n"] = out["sku_n"].fillna(0).astype(int)
out["active"] = out["freq"] >= 1

# Create frequency and SKU bins
out["freq_bin"] = np.where(out["freq"] > freq_cap, f">{freq_cap}", out["freq"].astype(str))
out["sku_bin"] = np.where(out["sku_n"] > sku_cap, f">{sku_cap}", out["sku_n"].astype(str))

# Normalize city and district names (handle case sensitivity)
if 'City' in out.columns and out['City'].notna().any():
    out['City_Normalized'] = out['City'].astype(str).str.strip().str.title()
    
if 'District' in out.columns and out['District'].notna().any():
    out['District_Normalized'] = out['District'].astype(str).str.strip().str.title()

# ============================ Determine Grouping Columns ============================
GROUPS = []
for c in ["Division", "AMO", "WH", "Channel", "Region_MNT", "Region", "SR"]:
    if c in out.columns and out[c].notna().any():
        GROUPS.append(c)

if not GROUPS:
    out["All"] = "All"
    GROUPS = ["All"]

# ============================ Summary Builder & Rankings ============================
@st.cache_data(ttl=300, show_spinner=False)
def build_summary(df: pd.DataFrame, groups: list[str], freq_cap: int, sku_cap: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(groups, dropna=False)
    
    # Basic counts
    mnt_count = g["outlet"].nunique().rename("Total_Outlets").reset_index()
    active_count = g["active"].sum().rename("Active_Outlets").reset_index()
    
    # Frequency distribution
    freq_ct = (g["freq_bin"].value_counts().rename("ct").reset_index()
               .pivot_table(index=groups, columns="freq_bin", values="ct", fill_value=0))
    
    # SKU distribution  
    sku_ct = (g["sku_bin"].value_counts().rename("ct").reset_index()
              .pivot_table(index=groups, columns="sku_bin", values="ct", fill_value=0))
    
    # SKU totals for average calculation
    sku_sum = g["sku_n"].sum().rename("Total_SKU").reset_index()
    
    # Create column lists
    freq_cols = [str(i) for i in range(freq_cap+1)] + [f">{freq_cap}"]
    sku_cols = [str(i) for i in range(sku_cap+1)] + [f">{sku_cap}"]
    
    # Build wide format
    wide = mnt_count.set_index(groups)
    
    # Add frequency columns
    for c in freq_cols:
        wide[f"Freq_{c}"] = freq_ct[c] if c in freq_ct.columns else 0
    
    # Add separator
    wide["|"] = ""
    
    # Add SKU columns
    for c in sku_cols:
        wide[f"SKU_{c}"] = sku_ct[c] if c in sku_ct.columns else 0
    
    # Convert back to dataframe and merge additional metrics
    wide = (wide.reset_index()
            .merge(active_count, on=groups, how="left")
            .merge(sku_sum, on=groups, how="left"))
    
    # Fix: Ensure active outlets never exceed total outlets
    wide["Active_Outlets"] = np.minimum(wide["Active_Outlets"], wide["Total_Outlets"])
    
    # Calculate percentages and averages with proper bounds checking
    wide["Outlet_Active_%"] = np.where(
        wide["Total_Outlets"] > 0, 
        np.minimum(100, (wide["Active_Outlets"] / wide["Total_Outlets"] * 100)).round(1), 
        0
    )
    wide["Avg_SKU_per_Outlet"] = np.where(wide["Total_Outlets"] > 0, 
                                          wide["Total_SKU"] / wide["Total_Outlets"], 0)
    
    return wide

def create_ranking_summary(df, group_col, title):
    """Create ranking summary for Division/AMO/Depo"""
    if group_col not in df.columns or df[group_col].isna().all():
        return pd.DataFrame()
    
    # Group and aggregate - fix the active calculation
    summary = df.groupby(group_col).agg({
        'outlet': 'nunique',  # Total outlets
        'active': 'sum',      # Total active outlets (sum of True/False converted to 1/0)
        'freq': 'mean',       # Average frequency
        'sku_n': 'mean'       # Average SKU count
    }).round(2)
    
    # Rename columns
    summary.columns = ['Total_Outlets', 'Active_Outlets', 'Avg_Frequency', 'Avg_SKU']
    
    # Fix: Ensure active outlets never exceed total outlets
    summary['Active_Outlets'] = np.minimum(summary['Active_Outlets'], summary['Total_Outlets'])
    
    # Calculate active rate with proper bounds checking
    summary['Active_Rate_%'] = np.where(
        summary['Total_Outlets'] > 0, 
        np.minimum(100, (summary['Active_Outlets'] / summary['Total_Outlets'] * 100)).round(1), 
        0
    )
    
    # Sort by active rate (highest first)
    summary = summary.sort_values('Active_Rate_%', ascending=False).reset_index()
    summary.index = summary.index + 1  # Start ranking from 1
    
    return summary

# ============================ Generate Summary ============================
summary = build_summary(out, GROUPS, freq_cap, sku_cap)

# ============================ Display Results with Tabs ============================
st.subheader("Outlet Analysis Dashboard")

if summary.empty:
    st.info("No data available for the current filters and date range.")
else:
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Overview", "🔍 Detailed Analysis", "🏆 Performance Rankings", "📈 Key Metrics"])
    
    with tab1:
        # Key insights only
        if not out.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            total_outlets = out["outlet"].nunique()
            active_outlets = (out["freq"] >= 1).sum()
            
            with col1:
                st.metric("Total Outlets", f"{total_outlets:,}")
                
            with col2:
                st.metric("Active Outlets", f"{active_outlets:,}", 
                         f"{100*active_outlets/total_outlets:.1f}%" if total_outlets > 0 else "0%")
            
            with col3:
                avg_freq = out[out["freq"] > 0]["freq"].mean() if (out["freq"] > 0).any() else 0
                st.metric("Avg Purchase Frequency", f"{avg_freq:.1f}")
                
            with col4:
                avg_sku = out[out["sku_n"] > 0]["sku_n"].mean() if (out["sku_n"] > 0).any() else 0
                st.metric("Avg SKU per Active Outlet", f"{avg_sku:.1f}")
        
        # Simplified summary table (key columns only)
        if not summary.empty:
            # Show only essential columns for overview
            key_cols = GROUPS + ["Total_Outlets", "Active_Outlets", "Outlet_Active_%", "Avg_SKU_per_Outlet"]
            available_key_cols = [col for col in key_cols if col in summary.columns]
            
            if available_key_cols:
                overview_summary = summary[available_key_cols].copy()
                
                # Format columns
                if "Outlet_Active_%" in overview_summary.columns:
                    overview_summary["Outlet_Active_%"] = overview_summary["Outlet_Active_%"].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "0%")
                if "Avg_SKU_per_Outlet" in overview_summary.columns:
                    overview_summary["Avg_SKU_per_Outlet"] = overview_summary["Avg_SKU_per_Outlet"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "0.0")
                
                st.subheader("Summary by Region/Organization")
                st.dataframe(overview_summary, use_container_width=True, height=400)
    
    with tab2:
        st.subheader("Purchase Frequency & SKU Distribution Analysis")
        
        # Prepare display columns for detailed analysis
        base_cols = GROUPS + ["Total_Outlets"]
        freq_cols = [f"Freq_{i}" for i in range(freq_cap+1)] + [f"Freq_>{freq_cap}"]
        sku_cols = [f"SKU_{i}" for i in range(sku_cap+1)] + [f"SKU_>{sku_cap}"]
        kpi_cols = ["Active_Outlets", "Outlet_Active_%", "Avg_SKU_per_Outlet"]
        
        # Ensure all columns exist
        for c in base_cols + freq_cols + ["|"] + sku_cols + kpi_cols:
            if c not in summary.columns:
                summary[c] = 0
        
        # Select and format display columns
        display_cols = base_cols + freq_cols + ["|"] + sku_cols + kpi_cols
        show = summary[display_cols].copy()
        
        # Format percentage and decimal columns
        if "Outlet_Active_%" in show.columns:
            show["Outlet_Active_%"] = show["Outlet_Active_%"].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "0%")
        if "Avg_SKU_per_Outlet" in show.columns:
            show["Avg_SKU_per_Outlet"] = show["Avg_SKU_per_Outlet"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "0.0")
        
        # Add grand total row
        if not out.empty:
            total_outlets = out["outlet"].nunique()
            active_outlets = (out["freq"] >= 1).sum()
            total_sku = out["sku_n"].sum()
            
            freq_counts = out["freq_bin"].value_counts()
            sku_counts = out["sku_bin"].value_counts()
            
            total_row = {c: "" for c in show.columns}
            total_row[GROUPS[-1]] = "TOTAL (All Filtered)"
            total_row["Total_Outlets"] = int(total_outlets)
            total_row["Active_Outlets"] = int(active_outlets)
            
            # Add frequency distribution to total
            for i in range(freq_cap+1):
                total_row[f"Freq_{i}"] = int(freq_counts.get(str(i), 0))
            total_row[f"Freq_>{freq_cap}"] = int(freq_counts.get(f">{freq_cap}", 0))
            
            # Add SKU distribution to total
            for i in range(sku_cap+1):
                total_row[f"SKU_{i}"] = int(sku_counts.get(str(i), 0))
            total_row[f"SKU_>{sku_cap}"] = int(sku_counts.get(f">{sku_cap}", 0))
            
            total_row["|"] = ""
            total_row["Outlet_Active_%"] = f"{100*active_outlets/total_outlets:.1f}%" if total_outlets > 0 else "0%"
            total_row["Avg_SKU_per_Outlet"] = f"{total_sku/total_outlets:.1f}" if total_outlets > 0 else "0.0"
            
            show = pd.concat([show.sort_values(base_cols), pd.DataFrame([total_row])], ignore_index=True)
        
        # Display the detailed table
        st.dataframe(show, use_container_width=True, height=500)
        
        # Explanation
        st.info(f"""
        **How to read this table:**
        - **Freq columns (0-{freq_cap}, >{freq_cap})**: Number of outlets with X purchase days in the selected date range
        - **SKU columns (0-{sku_cap}, >{sku_cap})**: Number of outlets buying X different products
        - **Active Rate %**: Percentage of outlets that made at least 1 purchase
        - **Avg SKU per Outlet**: Average number of different products per outlet
        """)
    
    with tab3:
        st.subheader("Performance Rankings")
        
        # Display ranking summaries - now sortable by users
        if show_division_summary and 'Division' in out.columns:
            st.write("**Division Performance Table** *(Click column headers to sort)*")
            div_ranking = create_ranking_summary(out, 'Division', 'Division')
            if not div_ranking.empty:
                st.dataframe(
                    div_ranking,
                    use_container_width=True,
                    column_config={
                        "Active_Rate_%": st.column_config.NumberColumn("Active Rate %", format="%.1f%%"),
                        "Avg_Frequency": st.column_config.NumberColumn("Avg Frequency", format="%.2f"),
                        "Avg_SKU": st.column_config.NumberColumn("Avg SKU", format="%.2f")
                    }
                )
            st.write("---")

        if show_amo_summary and 'AMO' in out.columns:
            st.write("**AMO Performance Table** *(Click column headers to sort)*")
            amo_ranking = create_ranking_summary(out, 'AMO', 'AMO')
            if not amo_ranking.empty:
                st.dataframe(
                    amo_ranking,
                    use_container_width=True,
                    column_config={
                        "Active_Rate_%": st.column_config.NumberColumn("Active Rate %", format="%.1f%%"),
                        "Avg_Frequency": st.column_config.NumberColumn("Avg Frequency", format="%.2f"),
                        "Avg_SKU": st.column_config.NumberColumn("Avg SKU", format="%.2f")
                    }
                )
            st.write("---")

        if show_depo_summary and 'WH' in out.columns:
            st.write("**Depo/Warehouse Performance Table** *(Click column headers to sort)*")
            depo_ranking = create_ranking_summary(out, 'WH', 'Depo/Warehouse')
            if not depo_ranking.empty:
                st.dataframe(
                    depo_ranking,
                    use_container_width=True,
                    column_config={
                        "Active_Rate_%": st.column_config.NumberColumn("Active Rate %", format="%.1f%%"),
                        "Avg_Frequency": st.column_config.NumberColumn("Avg Frequency", format="%.2f"),
                        "Avg_SKU": st.column_config.NumberColumn("Avg SKU", format="%.2f")
                    }
                )
        
        st.info("**Note**: All tables are sortable - click any column header to sort by that metric!")
    
    with tab4:
        st.subheader("Additional Analytics")
        
        if not out.empty:
            # Distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Purchase Frequency Distribution**")
                freq_dist = out["freq_bin"].value_counts().sort_index()
                st.bar_chart(freq_dist)
            
            with col2:
                st.write("**SKU Count Distribution**")
                sku_dist = out["sku_bin"].value_counts().sort_index()
                st.bar_chart(sku_dist)
            
            # City and District analysis
            city_col, district_col = st.columns(2)
            
            with city_col:
                if 'City_Normalized' in out.columns and out['City_Normalized'].notna().any():
                    st.write("**City Performance Overview**")
                    city_summary = out.groupby('City_Normalized').agg({
                        'outlet': 'nunique',
                        'active': 'sum',
                        'freq': 'mean',
                        'sku_n': 'mean'
                    }).round(2)
                    city_summary.columns = ['Total_Outlets', 'Active_Outlets', 'Avg_Frequency', 'Avg_SKU']
                    city_summary['Active_Rate_%'] = (city_summary['Active_Outlets'] / city_summary['Total_Outlets'] * 100).round(1)
                    
                    # Sort by total outlets (descending) to show major cities first
                    city_summary = city_summary.sort_values('Total_Outlets', ascending=False)
                    
                    st.dataframe(
                        city_summary,
                        use_container_width=True,
                        column_config={
                            "Active_Rate_%": st.column_config.NumberColumn("Active Rate %", format="%.1f%%"),
                            "Avg_Frequency": st.column_config.NumberColumn("Avg Frequency", format="%.2f"),
                            "Avg_SKU": st.column_config.NumberColumn("Avg SKU", format="%.2f")
                        }
                    )
                else:
                    st.info("City data not available")
            
            with district_col:
                if 'District_Normalized' in out.columns and out['District_Normalized'].notna().any():
                    st.write("**District Performance Overview**")
                    district_summary = out.groupby('District_Normalized').agg({
                        'outlet': 'nunique',
                        'active': 'sum',
                        'freq': 'mean',
                        'sku_n': 'mean'
                    }).round(2)
                    district_summary.columns = ['Total_Outlets', 'Active_Outlets', 'Avg_Frequency', 'Avg_SKU']
                    district_summary['Active_Rate_%'] = (district_summary['Active_Outlets'] / district_summary['Total_Outlets'] * 100).round(1)
                    
                    # Sort by total outlets (descending) to show major districts first
                    district_summary = district_summary.sort_values('Total_Outlets', ascending=False)
                    
                    st.dataframe(
                        district_summary,
                        use_container_width=True,
                        column_config={
                            "Active_Rate_%": st.column_config.NumberColumn("Active Rate %", format="%.1f%%"),
                            "Avg_Frequency": st.column_config.NumberColumn("Avg Frequency", format="%.2f"),
                            "Avg_SKU": st.column_config.NumberColumn("Avg SKU", format="%.2f")
                        }
                    )
                else:
                    st.info("District data not available")
            
            st.caption("*City and District names are normalized to handle case sensitivity (e.g., 'cengkareng', 'CENGKARENG', 'Cengkareng' are treated as the same)*")

# ============================ Downloads ============================
st.subheader("Download Reports")

col1, col2 = st.columns(2)

with col1:
    if not summary.empty:
        csv_data = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📊 Download Summary CSV",
            data=csv_data,
            file_name=f"outlet_analysis_summary_{d1}_{d2}.csv",
            mime="text/csv"
        )

with col2:
    # Excel download with multiple sheets including rankings
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not summary.empty:
            summary.to_excel(writer, index=False, sheet_name="Summary")
        if not out.empty:
            out.to_excel(writer, index=False, sheet_name="Outlet_Detail")
        if not m.empty:
            m.to_excel(writer, index=False, sheet_name="Outlet_Master")
        
        # Add ranking sheets
        if show_division_summary and 'Division' in out.columns:
            div_ranking = create_ranking_summary(out, 'Division', 'Division')
            if not div_ranking.empty:
                div_ranking.to_excel(writer, index=True, sheet_name="Division_Ranking")
        
        if show_amo_summary and 'AMO' in out.columns:
            amo_ranking = create_ranking_summary(out, 'AMO', 'AMO')
            if not amo_ranking.empty:
                amo_ranking.to_excel(writer, index=True, sheet_name="AMO_Ranking")
        
        if show_depo_summary and 'WH' in out.columns:
            depo_ranking = create_ranking_summary(out, 'WH', 'Depo/Warehouse')
            if not depo_ranking.empty:
                depo_ranking.to_excel(writer, index=True, sheet_name="Depo_Ranking")
    
    st.download_button(
        "📋 Download Complete Excel Report",
        data=buf.getvalue(),
        file_name=f"outlet_management_report_{d1}_{d2}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================ Debug Information ============================
with st.expander("🔧 Debug: Schema Detection & Configuration"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Table Configuration:**")
        st.json({
            "Active Sales Table": T_SALE,
            "USE_DAKAR": USE_DAKAR,
            "Date Range": f"{d1} to {d2}",
            "Frequency Cap": f">{freq_cap}",
            "SKU Cap": f">{sku_cap}"
        })
    
    with col2:
        st.write("**Column Mappings:**")
        st.json({
            "MNT Columns": {
                "Customer": CUST_O,
                "Region": REG_O,
                "Warehouse": WH_O,
                "Division": DIV_O,
                "AMO": AMO_O,
                "Channel": CH_O
            },
            "DAKAR Columns": {
                "Division": DIV_D,
                "AMO": AMO_D,
                "Warehouse": WH_D,
                "Channel": CH_D,
                "Region": REG_D,
                "SR": SR_D
            },
            "Sales Columns": {
                "Date": DATE_S,
                "Customer": CUST_S,
                "SKU": SKU_S
            }
        })
    
    if not out.empty:
        st.write("**Data Sample:**")
        st.dataframe(out.head(10), use_container_width=True)
