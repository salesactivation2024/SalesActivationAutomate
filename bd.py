import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from io import BytesIO
import plotly.graph_objects as go

import datetime as dt
import calendar
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text as sa_text
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

st.set_page_config(page_title="Customer Distribution by Brand", layout="wide")

# ---------------- DB connection ----------------
@st.cache_resource
def make_engine():
    try:
        s = st.secrets["dbo"]
        
        # Try different ODBC drivers in order of preference
        drivers_to_try = [
            "ODBC Driver 17 for SQL Server",
            "ODBC Driver 13 for SQL Server", 
            "ODBC Driver 11 for SQL Server",
            "SQL Server Native Client 11.0",
            "SQL Server"
        ]
        
        engine = None
        last_error = None
        
        for driver in drivers_to_try:
            try:
                odbc_str = (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={s['host']},{s['port']};DATABASE={s['database']};UID={s['username']};PWD={s['password']};"
                    "Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
                )
                url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str)
                engine = create_engine(url)
                
                # Test the connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                st.sidebar.success(f"Connected using: {driver}")
                return engine
                
            except Exception as e:
                last_error = e
                continue
        
        # If all drivers fail, show error
        raise Exception(f"Could not connect with any ODBC driver. Last error: {str(last_error)}")
        
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.info("Please check your database configuration in Streamlit secrets.")
        st.stop()

engine = make_engine()

# Define table names
T_SALE = st.secrets.get("tables", {}).get("smsdv", "dbo.smsdv")
T_MNT = st.secrets.get("tables", {}).get("mnt", "dbo.mnt")
T_DAKAR = st.secrets.get("tables", {}).get("dakar", "dbo.dakar")

# ---------------- Helper Functions ----------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_columns(table: str):
    q = text("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = PARSENAME(:t,1)
          AND (TABLE_SCHEMA = PARSENAME(:t,2) OR :t NOT LIKE '%.%')
        ORDER BY ORDINAL_POSITION
    """)
    return pd.read_sql(q, engine, params={"t": table})["COLUMN_NAME"].str.strip().tolist()

def pick(available, *cands):
    if not available: return None
    low = {c.lower(): c for c in available}
    for cand in cands:
        if cand and cand.lower() in low:
            return low[cand.lower()]
    return None

# Get columns
ALL_SALE = get_columns(T_SALE)
ALL_MNT = get_columns(T_MNT)
ALL_DAKAR = get_columns(T_DAKAR)

# Map columns - SALES
DATE_S = pick(ALL_SALE, "converted_date", "Sales_Date", "Tanggal", "Date")
QTY_S = pick(ALL_SALE, "Sales_Qty_", "Qty", "Quantity")
PRICE_S = pick(ALL_SALE, "Sales_Price", "Sales_Amount", "Total_Sales", "Price")
SKU_S = pick(ALL_SALE, "Prod_Name", "Product_Name", "SKU", "SKU_Name")
CUST_S = pick(ALL_SALE, "Customer_Code", "Cust_Code", "Outlet_Code", "Customer")
REG_S = pick(ALL_SALE, "Region", "Region_Code", "Area")
WS_S = pick(ALL_SALE, "Ws_Name", "WH_Name", "Warehouse", "From_W_H")
CH_S = pick(ALL_SALE, "Channel_Low_", "Channel", "Channel_Low")

# Map columns - MNT
CUST_M = pick(ALL_MNT, "Customer", "Customer_Code", "Cust_Code", "Outlet_Code")
REG_M = pick(ALL_MNT, "Region", "Region_Code", "Area")
WS_M = pick(ALL_MNT, "Ws_Name", "WH_Name", "Warehouse", "From_W_H")
CH_M = pick(ALL_MNT, "Channel", "Channel_Low_", "Channel_Low")

# Map columns - DAKAR
REG_D = pick(ALL_DAKAR, "Region", "Region_Code")
DIV_D = pick(ALL_DAKAR, "Division", "Divisi")
AMO_D = pick(ALL_DAKAR, "AMO", "Sales_Office_Name")
DEPO_D = pick(ALL_DAKAR, "Depo", "WH_Name", "Warehouse")

# Get date bounds
@st.cache_data(ttl=900, show_spinner=False)
def get_date_bounds():
    if not DATE_S:
        today = dt.date.today()
        return today - dt.timedelta(days=30), today
    
    # Check if we're using converted_date (datetime) or Sales_Date (bigint)
    if DATE_S == "converted_date":
        q = text(f"SELECT CAST(MIN({DATE_S}) AS date) mn, CAST(MAX({DATE_S}) AS date) mx FROM {T_SALE} WHERE {DATE_S} IS NOT NULL")
    else:
        q = text(f"SELECT MIN({DATE_S}) as mn, MAX({DATE_S}) as mx FROM {T_SALE} WHERE {DATE_S} IS NOT NULL")
    
    r = pd.read_sql(q, engine).iloc[0]
    mn, mx = r["mn"], r["mx"]
    
    if pd.isna(mn) or pd.isna(mx):
        today = dt.date.today()
        return today - dt.timedelta(days=30), today
    
    if DATE_S != "converted_date":
        # Convert bigint YYYYMMDD to date
        try:
            mn = dt.datetime.strptime(str(int(mn)), '%Y%m%d').date()
            mx = dt.datetime.strptime(str(int(mx)), '%Y%m%d').date()
        except:
            today = dt.date.today()
            return today - dt.timedelta(days=30), today
    
    return mn, mx

# Get distinct values
@st.cache_data(ttl=900, show_spinner=False)
def get_distinct_values(table, col):
    if not col: return []
    
    if table == "mnt":
        sql = text(f"SELECT DISTINCT {col} AS v FROM {T_MNT} WHERE {col} IS NOT NULL ORDER BY v")
    elif table == "dakar":
        sql = text(f"SELECT DISTINCT {col} AS v FROM {T_DAKAR} WHERE {col} IS NOT NULL ORDER BY v")
    else:
        sql = text(f"SELECT DISTINCT {col} AS v FROM {T_SALE} WHERE {col} IS NOT NULL ORDER BY v")
    
    return pd.read_sql(sql, engine)["v"].astype(str).tolist()

# ---------------- Page Selection ----------------
page = st.sidebar.radio("📄 Select Page", ["Distribution Analysis", "Weekly Comparison"], index=0)

# ---------------- Sidebar Filters ----------------
st.sidebar.header("🔎 Filters")

# Date range
dmin, dmax = get_date_bounds()
date_range = st.sidebar.date_input("Sales Date Range", (dmin, dmax), min_value=dmin, max_value=dmax)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = date_range
elif isinstance(date_range, list) and len(date_range) == 2:
    d1, d2 = date_range[0], date_range[1]
else:
    d1 = d2 = date_range if not isinstance(date_range, (list, tuple)) else dmin

if d1 > d2:
    d1, d2 = d2, d1

# Get filter options from DAKAR and MNT
divisions = get_distinct_values("dakar", DIV_D) if DIV_D else []
amos = get_distinct_values("dakar", AMO_D) if AMO_D else []
regions = get_distinct_values("mnt", REG_M)
warehouses = get_distinct_values("mnt", WS_M)
channels = get_distinct_values("mnt", CH_M)

# Filters
div_sel = st.sidebar.multiselect("Division (from DAKAR)", divisions, default=[])
amo_sel = st.sidebar.multiselect("AMO (from DAKAR)", amos, default=[])
reg_sel = st.sidebar.multiselect("Region", regions, default=[])
ws_sel = st.sidebar.multiselect("Warehouse", warehouses, default=[])
ch_sel = st.sidebar.multiselect("Channel", channels, default=[])

# Display options
group_by = st.sidebar.radio("Group By", ["Kategori (Brand)", "SKU"], index=0)
use_kategori = group_by == "Kategori (Brand)"

st.sidebar.write("---")
st.sidebar.info(f"📅 Using date column: {DATE_S}")

# ---------------- Load Distribution Data Function ----------------
@st.cache_data(ttl=600, show_spinner=True)
def load_distribution_data(d1, d2, div_sel, amo_sel, reg_sel, ws_sel, ch_sel, use_kategori):
    
    # Build MNT query with DAKAR join for Division/AMO filtering
    mnt_conditions = []
    mnt_params = {}
    
    # Join MNT with DAKAR for Division and AMO filtering
    mnt_base = f"""
    FROM {T_MNT} m
    LEFT JOIN {T_DAKAR} d ON m.{REG_M} = d.{REG_D}
    WHERE 1=1
    """
    
    if div_sel and DIV_D:
        div_placeholders = ','.join([f':div{i}' for i in range(len(div_sel))])
        mnt_conditions.append(f"d.{DIV_D} IN ({div_placeholders})")
        for i, div in enumerate(div_sel):
            mnt_params[f'div{i}'] = div
    
    if amo_sel and AMO_D:
        amo_placeholders = ','.join([f':amo{i}' for i in range(len(amo_sel))])
        mnt_conditions.append(f"d.{AMO_D} IN ({amo_placeholders})")
        for i, amo in enumerate(amo_sel):
            mnt_params[f'amo{i}'] = amo
    
    if reg_sel and REG_M:
        reg_placeholders = ','.join([f':reg{i}' for i in range(len(reg_sel))])
        mnt_conditions.append(f"m.{REG_M} IN ({reg_placeholders})")
        for i, reg in enumerate(reg_sel):
            mnt_params[f'reg{i}'] = reg
    
    if ws_sel and WS_M:
        ws_placeholders = ','.join([f':ws{i}' for i in range(len(ws_sel))])
        mnt_conditions.append(f"m.{WS_M} IN ({ws_placeholders})")
        for i, ws in enumerate(ws_sel):
            mnt_params[f'ws{i}'] = ws
    
    if ch_sel and CH_M:
        ch_placeholders = ','.join([f':ch{i}' for i in range(len(ch_sel))])
        mnt_conditions.append(f"m.{CH_M} IN ({ch_placeholders})")
        for i, ch in enumerate(ch_sel):
            mnt_params[f'ch{i}'] = ch
    
    if mnt_conditions:
        mnt_base += " AND " + " AND ".join(mnt_conditions)
    
    # Get total customers by channel (RO)
    mnt_query = f"""
    SELECT 
        m.{CH_M if CH_M else "'ALL'"} as Channel,
        COUNT(DISTINCT m.{CUST_M}) as RO
    {mnt_base}
    GROUP BY m.{CH_M if CH_M else "'ALL'"}
    """
    
    df_mnt = pd.read_sql(text(mnt_query), engine, params=mnt_params)
    
    # Also get total for ALL CHANNEL
    total_query = f"""
    SELECT 
        'ALL CHANNEL' as Channel,
        COUNT(DISTINCT m.{CUST_M}) as RO
    {mnt_base}
    """
    
    df_total = pd.read_sql(text(total_query), engine, params=mnt_params)
    df_mnt = pd.concat([df_mnt, df_total], ignore_index=True)
    
    # Build SALES query conditions
    if DATE_S == "converted_date":
        sale_conditions = [f"CAST(s.{DATE_S} AS date) BETWEEN :d1 AND :d2"]
        sale_params = {"d1": d1, "d2": d2}
    else:
        d1_int = int(d1.strftime('%Y%m%d'))
        d2_int = int(d2.strftime('%Y%m%d'))
        sale_conditions = [f"s.{DATE_S} BETWEEN :d1 AND :d2"]
        sale_params = {"d1": d1_int, "d2": d2_int}
    
    # Add filter conditions via MNT-DAKAR join
    sales_join = f"""
    FROM {T_SALE} s
    INNER JOIN {T_MNT} m ON s.{CUST_S} = m.{CUST_M}
    LEFT JOIN {T_DAKAR} d ON m.{REG_M} = d.{REG_D}
    """
    
    if div_sel and DIV_D:
        div_placeholders = ','.join([f':sdiv{i}' for i in range(len(div_sel))])
        sale_conditions.append(f"d.{DIV_D} IN ({div_placeholders})")
        for i, div in enumerate(div_sel):
            sale_params[f'sdiv{i}'] = div
    
    if amo_sel and AMO_D:
        amo_placeholders = ','.join([f':samo{i}' for i in range(len(amo_sel))])
        sale_conditions.append(f"d.{AMO_D} IN ({amo_placeholders})")
        for i, amo in enumerate(amo_sel):
            sale_params[f'samo{i}'] = amo
    
    if reg_sel and REG_S:
        reg_placeholders = ','.join([f':sreg{i}' for i in range(len(reg_sel))])
        sale_conditions.append(f"s.{REG_S} IN ({reg_placeholders})")
        for i, reg in enumerate(reg_sel):
            sale_params[f'sreg{i}'] = reg
    
    if ws_sel and WS_S:
        ws_placeholders = ','.join([f':sws{i}' for i in range(len(ws_sel))])
        sale_conditions.append(f"s.{WS_S} IN ({ws_placeholders})")
        for i, ws in enumerate(ws_sel):
            sale_params[f'sws{i}'] = ws
    
    if ch_sel and CH_S:
        ch_placeholders = ','.join([f':sch{i}' for i in range(len(ch_sel))])
        sale_conditions.append(f"s.{CH_S} IN ({ch_placeholders})")
        for i, ch in enumerate(ch_sel):
            sale_params[f'sch{i}'] = ch
    
    where_sale = " WHERE " + " AND ".join(sale_conditions)
    
    if use_kategori:
        # Query for Kategori grouping
        kategori_query = f"""
        SELECT 
            UPPER(SUBSTRING(s.{SKU_S}, 1, CHARINDEX(' ', s.{SKU_S} + ' ') - 1)) as Kategori,
            s.{CH_S if CH_S else "'ALL'"} as Channel,
            COUNT(DISTINCT s.{CUST_S}) as Dist,
            SUM(s.{QTY_S if QTY_S else '1'}) as Total_Qty,
            SUM(s.{PRICE_S if PRICE_S else '0'}) as Total_Value
        {sales_join}
        {where_sale}
        GROUP BY UPPER(SUBSTRING(s.{SKU_S}, 1, CHARINDEX(' ', s.{SKU_S} + ' ') - 1)), s.{CH_S if CH_S else "'ALL'"}
        """
        
        df_sales = pd.read_sql(text(kategori_query), engine, params=sale_params)
        
        # Get ALL CHANNEL totals for kategori
        kategori_total_query = f"""
        SELECT 
            UPPER(SUBSTRING(s.{SKU_S}, 1, CHARINDEX(' ', s.{SKU_S} + ' ') - 1)) as Kategori,
            'ALL CHANNEL' as Channel,
            COUNT(DISTINCT s.{CUST_S}) as Dist,
            SUM(s.{QTY_S if QTY_S else '1'}) as Total_Qty,
            SUM(s.{PRICE_S if PRICE_S else '0'}) as Total_Value
        {sales_join}
        {where_sale}
        GROUP BY UPPER(SUBSTRING(s.{SKU_S}, 1, CHARINDEX(' ', s.{SKU_S} + ' ') - 1))
        """
        
        df_total_sales = pd.read_sql(text(kategori_total_query), engine, params=sale_params)
        df_sales = pd.concat([df_sales, df_total_sales], ignore_index=True)
        
        # Get SKU list for each kategori
        sku_list_query = f"""
        SELECT DISTINCT
            UPPER(SUBSTRING(s.{SKU_S}, 1, CHARINDEX(' ', s.{SKU_S} + ' ') - 1)) as Kategori,
            s.{SKU_S} as SKU
        {sales_join}
        {where_sale}
        """
        
        df_sku_list = pd.read_sql(text(sku_list_query), engine, params=sale_params)
        sku_grouped = df_sku_list.groupby('Kategori')['SKU'].apply(lambda x: '<br>'.join(x.unique()[:5])).reset_index()
        sku_grouped.rename(columns={'SKU': 'SKU_List'}, inplace=True)
        
        df_sales = df_sales.merge(sku_grouped, on='Kategori', how='left')
        df_sales.rename(columns={'Kategori': 'Item'}, inplace=True)
    else:
        # Query for SKU grouping
        sales_query = f"""
        SELECT 
            s.{SKU_S} as Item,
            s.{CH_S if CH_S else "'ALL'"} as Channel,
            COUNT(DISTINCT s.{CUST_S}) as Dist,
            SUM(s.{QTY_S if QTY_S else '1'}) as Total_Qty,
            SUM(s.{PRICE_S if PRICE_S else '0'}) as Total_Value
        {sales_join}
        {where_sale}
        GROUP BY s.{SKU_S}, s.{CH_S if CH_S else "'ALL'"}
        """
        
        df_sales = pd.read_sql(text(sales_query), engine, params=sale_params)
        
        # Get ALL CHANNEL totals
        total_sales_query = f"""
        SELECT 
            s.{SKU_S} as Item,
            'ALL CHANNEL' as Channel,
            COUNT(DISTINCT s.{CUST_S}) as Dist,
            SUM(s.{QTY_S if QTY_S else '1'}) as Total_Qty,
            SUM(s.{PRICE_S if PRICE_S else '0'}) as Total_Value
        {sales_join}
        {where_sale}
        GROUP BY s.{SKU_S}
        """
        
        df_total_sales = pd.read_sql(text(total_sales_query), engine, params=sale_params)
        df_sales = pd.concat([df_sales, df_total_sales], ignore_index=True)
        df_sales['SKU_List'] = df_sales['Item']
    
    return df_mnt, df_sales

# ---------------- Weekly Comparison Function ----------------
@st.cache_data(ttl=600, show_spinner=True)
def load_weekly_distribution(d1, d2, div_sel, amo_sel, reg_sel, ws_sel, ch_sel, selected_skus):
    # Get total RO
    mnt_base = f"""
    FROM {T_MNT} m
    LEFT JOIN {T_DAKAR} d ON m.{REG_M} = d.{REG_D}
    WHERE 1=1
    """
    
    mnt_conditions = []
    mnt_params = {}
    
    if div_sel and DIV_D:
        div_placeholders = ','.join([f':div{i}' for i in range(len(div_sel))])
        mnt_conditions.append(f"d.{DIV_D} IN ({div_placeholders})")
        for i, div in enumerate(div_sel):
            mnt_params[f'div{i}'] = div
    
    if amo_sel and AMO_D:
        amo_placeholders = ','.join([f':amo{i}' for i in range(len(amo_sel))])
        mnt_conditions.append(f"d.{AMO_D} IN ({amo_placeholders})")
        for i, amo in enumerate(amo_sel):
            mnt_params[f'amo{i}'] = amo
    
    if mnt_conditions:
        mnt_base += " AND " + " AND ".join(mnt_conditions)
    
    total_ro_query = f"SELECT COUNT(DISTINCT m.{CUST_M}) as Total_RO {mnt_base}"
    total_ro = pd.read_sql(text(total_ro_query), engine, params=mnt_params)['Total_RO'].iloc[0]
    
    # Build sales query for weekly data
    sales_join = f"""
    FROM {T_SALE} s
    INNER JOIN {T_MNT} m ON s.{CUST_S} = m.{CUST_M}
    LEFT JOIN {T_DAKAR} d ON m.{REG_M} = d.{REG_D}
    """
    
    if DATE_S == "converted_date":
        sale_conditions = [f"CAST(s.{DATE_S} AS date) BETWEEN :d1 AND :d2"]
        sale_params = {"d1": d1, "d2": d2}
        date_col = f"CAST(s.{DATE_S} AS date)"
    else:
        d1_int = int(d1.strftime('%Y%m%d'))
        d2_int = int(d2.strftime('%Y%m%d'))
        sale_conditions = [f"s.{DATE_S} BETWEEN :d1 AND :d2"]
        sale_params = {"d1": d1_int, "d2": d2_int}
        date_col = f"CAST(CAST(s.{DATE_S} as VARCHAR(8)) as DATE)"
    
    if selected_skus and SKU_S:
        sku_placeholders = ','.join([f':sku{i}' for i in range(len(selected_skus))])
        sale_conditions.append(f"s.{SKU_S} IN ({sku_placeholders})")
        for i, sku in enumerate(selected_skus):
            sale_params[f'sku{i}'] = sku
    
    if div_sel and DIV_D:
        div_placeholders = ','.join([f':sdiv{i}' for i in range(len(div_sel))])
        sale_conditions.append(f"d.{DIV_D} IN ({div_placeholders})")
        for i, div in enumerate(div_sel):
            sale_params[f'sdiv{i}'] = div
    
    if amo_sel and AMO_D:
        amo_placeholders = ','.join([f':samo{i}' for i in range(len(amo_sel))])
        sale_conditions.append(f"d.{AMO_D} IN ({amo_placeholders})")
        for i, amo in enumerate(amo_sel):
            sale_params[f'samo{i}'] = amo
    
    where_sale = " WHERE " + " AND ".join(sale_conditions)
    
    weekly_query = f"""
    SELECT 
        s.{SKU_S} as SKU,
        DATEPART(YEAR, {date_col}) as Year,
        DATEPART(WEEK, {date_col}) as Week,
        MIN({date_col}) as Week_Start,
        MAX({date_col}) as Week_End,
        COUNT(DISTINCT s.{CUST_S}) as Dist
    {sales_join}
    {where_sale}
    GROUP BY s.{SKU_S}, DATEPART(YEAR, {date_col}), DATEPART(WEEK, {date_col})
    ORDER BY Year, Week
    """
    
    df_weekly = pd.read_sql(text(weekly_query), engine, params=sale_params)
    
    if not df_weekly.empty:
        df_weekly['Distribution_%'] = (df_weekly['Dist'] / total_ro * 100).round(2)
        df_weekly['Week_Label'] = 'W' + df_weekly['Week'].astype(str) + '-' + df_weekly['Year'].astype(str).str[-2:]
    
    return df_weekly, total_ro

# ---------------- Main Application ----------------
st.title("📊 Customer Distribution by Brand/Kategori")

if page == "Distribution Analysis":
    # Load data
    with st.spinner("Loading distribution data..."):
        df_mnt, df_sales = load_distribution_data(d1, d2, div_sel, amo_sel, reg_sel, ws_sel, ch_sel, use_kategori)
    
    if df_sales.empty:
        st.warning("No sales data found for the selected filters.")
        st.stop()
    
    # Merge with MNT to get RO values
    df_final = df_sales.merge(df_mnt, on='Channel', how='left')
    df_final['RO'] = df_final['RO'].fillna(0)
    df_final['Dist'] = df_final['Dist'].fillna(0)
    df_final['Distribution_%'] = np.where(df_final['RO'] > 0, (df_final['Dist'] / df_final['RO'] * 100).round(1), 0)
    
    # Get unique channels
    channels_list = [ch for ch in df_final['Channel'].unique() if ch != 'ALL CHANNEL']
    channels_list.sort()
    channels_list.append('ALL CHANNEL')
    
    # Create pivot table
    pivot_data = []
    for item in df_final['Item'].unique():
        row = {'Item': item, 'SKU': df_final[df_final['Item'] == item]['SKU_List'].iloc[0] if 'SKU_List' in df_final.columns else item}
        
        for ch in channels_list:
            ch_data = df_final[(df_final['Item'] == item) & (df_final['Channel'] == ch)]
            
            if not ch_data.empty:
                ro_value = ch_data['RO'].iloc[0]
                dist_value = ch_data['Dist'].iloc[0]
                pct_value = ch_data['Distribution_%'].iloc[0]
                
                row[f'{ch}_RO'] = int(ro_value) if pd.notna(ro_value) else 0
                row[f'{ch}_Dist'] = int(dist_value) if pd.notna(dist_value) else 0
                row[f'{ch}_%'] = f"{pct_value:.1f}%" if pd.notna(pct_value) else "0%"
            else:
                row[f'{ch}_RO'] = 0
                row[f'{ch}_Dist'] = 0
                row[f'{ch}_%'] = "0%"
        
        pivot_data.append(row)
    
    df_pivot = pd.DataFrame(pivot_data)
    
    # Sort by ALL CHANNEL distribution
    if 'ALL CHANNEL_Dist' in df_pivot.columns:
        df_pivot = df_pivot.sort_values('ALL CHANNEL_Dist', ascending=False)
    
    # Display results
    st.subheader(f"📊 Distribution by {group_by}")
    
    # Display columns
    display_columns = ['Item', 'SKU'] if use_kategori else ['Item']
    for ch in channels_list:
        if f'{ch}_RO' in df_pivot.columns:
            display_columns.extend([f'{ch}_RO', f'{ch}_Dist', f'{ch}_%'])
    
    display_columns = [col for col in display_columns if col in df_pivot.columns]
    
    st.dataframe(df_pivot[display_columns], use_container_width=True, height=500)
    
    # Summary metrics
    st.write("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_items = df_pivot['Item'].nunique()
        st.metric(f"Total {group_by}", f"{total_items:,}")
    
    with col2:
        if 'ALL CHANNEL_RO' in df_pivot.columns:
            total_ro = df_pivot['ALL CHANNEL_RO'].iloc[0] if not df_pivot.empty and pd.notna(df_pivot['ALL CHANNEL_RO'].iloc[0]) else 0
            st.metric("Total Customers (RO)", f"{total_ro:,}")
    
    with col3:
        if 'ALL CHANNEL_Dist' in df_pivot.columns:
            total_dist = df_pivot['ALL CHANNEL_Dist'].sum()
            st.metric("Total Active Customers", f"{total_dist:,}")
    
    with col4:
        if 'ALL CHANNEL_RO' in df_pivot.columns and 'ALL CHANNEL_Dist' in df_pivot.columns:
            total_ro = df_pivot['ALL CHANNEL_RO'].iloc[0] if not df_pivot.empty and pd.notna(df_pivot['ALL CHANNEL_RO'].iloc[0]) else 0
            total_dist = df_pivot['ALL CHANNEL_Dist'].sum()
            avg_dist = (total_dist / total_ro * 100) if total_ro > 0 else 0
            st.metric("Overall Distribution %", f"{avg_dist:.1f}%")
    
    # Download options
    st.write("---")
    st.subheader("📥 Download Options")
    
    csv = df_pivot[display_columns].to_csv(index=False).encode('utf-8')
    st.download_button(
        "📊 Download Distribution CSV",
        data=csv,
        file_name=f"distribution_{group_by}_{d1}_{d2}.csv",
        mime="text/csv"
    )

else:  # Weekly Comparison page
    st.subheader("📈 Weekly Distribution Comparison")
    
    # Get available SKUs
    @st.cache_data(ttl=300)
    def get_available_skus(d1, d2):
        if DATE_S == "converted_date":
            cond = f"CAST({DATE_S} AS date) BETWEEN :d1 AND :d2"
            params = {"d1": d1, "d2": d2}
        else:
            d1_int = int(d1.strftime('%Y%m%d'))
            d2_int = int(d2.strftime('%Y%m%d'))
            cond = f"{DATE_S} BETWEEN :d1 AND :d2"
            params = {"d1": d1_int, "d2": d2_int}
        
        q = text(f"SELECT DISTINCT {SKU_S} as SKU FROM {T_SALE} WHERE {cond} ORDER BY {SKU_S}")
        return pd.read_sql(q, engine, params=params)["SKU"].tolist()
    
    available_skus = get_available_skus(d1, d2)
    selected_skus = st.multiselect("Select SKUs for Comparison", available_skus, default=available_skus[:5])
    
    if not selected_skus:
        st.warning("Please select at least one SKU to view weekly comparisons.")
        st.stop()
    
    # Load weekly data
    with st.spinner("Loading weekly distribution data..."):
        df_weekly, total_ro = load_weekly_distribution(d1, d2, div_sel, amo_sel, reg_sel, ws_sel, ch_sel, selected_skus)
    
    if df_weekly.empty:
        st.warning("No data found for the selected filters and SKUs.")
        st.stop()
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total RO", f"{total_ro:,}")
    
    with col2:
        st.metric("Selected SKUs", len(selected_skus))
    
    with col3:
        st.metric("Date Range", f"{d1} to {d2}")
    
    with col4:
        weeks_count = df_weekly['Week_Label'].nunique()
        st.metric("Weeks Covered", weeks_count)
    
    # Create pivot table
    st.subheader("📅 Weekly Distribution Trend")
    
    pivot_weekly = df_weekly.pivot_table(
        index='SKU',
        columns='Week_Label',
        values='Distribution_%',
        fill_value=0
    )
    
    if not pivot_weekly.empty:
        pivot_weekly['Average'] = pivot_weekly.mean(axis=1).round(2)
        
        # Format for display
        display_weekly = pivot_weekly.copy()
        for col in display_weekly.columns:
            if col != 'Average':
                display_weekly[col] = display_weekly[col].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        
        st.dataframe(display_weekly, use_container_width=True, height=400)
        
        # Line chart
        st.subheader("📈 Distribution Trend Chart")
        
        fig = go.Figure()
        
        for sku in selected_skus[:10]:  # Limit to 10 SKUs
            sku_data = df_weekly[df_weekly['SKU'] == sku].sort_values(['Year', 'Week'])
            if not sku_data.empty:
                fig.add_trace(go.Scatter(
                    x=sku_data['Week_Label'],
                    y=sku_data['Distribution_%'],
                    mode='lines+markers',
                    name=sku[:30],
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Weekly Distribution % Trend",
            xaxis_title="Week",
            yaxis_title="Distribution %",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.write("---")
        csv_weekly = pivot_weekly.to_csv().encode('utf-8')
        st.download_button(
            "📊 Download Weekly CSV",
            data=csv_weekly,
            file_name=f"weekly_distribution_{d1}_{d2}.csv",
            mime="text/csv"

        )
