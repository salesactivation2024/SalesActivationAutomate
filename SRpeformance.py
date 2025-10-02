import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from io import BytesIO

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

# ============================ App config ============================
st.set_page_config(page_title="CM/EC Dashboard (DB)", layout="wide")
st.title("🏅 SR Performance Dashboard")
st.caption("Data is queried directly from SQL Server. Filters apply server-side. Hari Kerja = actual days with sales (Sat weighted, Sun=0).")

# ============================ Connection ============================
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

# Table names (override in secrets if needed)
T_CMEC  = st.secrets.get("tables", {}).get("cmec",  "dbo.cmec")
T_DAKAR = st.secrets.get("tables", {}).get("dakar", "dbo.dakar")
T_MNT   = st.secrets.get("tables", {}).get("mnt",   "dbo.mnt")

# ============================ Helpers ============================
@st.cache_data(ttl=900, show_spinner=False)
def get_columns(table: str):
    q = text("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE (TABLE_SCHEMA = PARSENAME(:t,2) AND TABLE_NAME = PARSENAME(:t,1))
              OR TABLE_NAME = :t
    """)
    return pd.read_sql(q, engine, params={"t": table})["COLUMN_NAME"].str.strip().tolist()

@st.cache_data(ttl=900, show_spinner=False)
def distinct_vals(table, col):
    try:
        q = text(f"SELECT DISTINCT {col} AS v FROM {table} WHERE {col} IS NOT NULL ORDER BY v")
        return pd.read_sql(q, engine)["v"].astype(str).tolist()
    except:
        return []

def pct_series(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den * 100).round(1).fillna(0)

# ============================ Sidebar Filters ============================
with st.sidebar:
    st.subheader("📊 Filters")
    today = dt.date.today()
    d1, d2 = st.date_input("Sales Date Range", (dt.date(today.year, 1, 1), today))

    divisions = distinct_vals(T_DAKAR, "Division")
    amos      = distinct_vals(T_DAKAR, "AMO")
    depos     = distinct_vals(T_DAKAR, "Depo")
    channels  = distinct_vals(T_DAKAR, "Channel")
    regions   = distinct_vals(T_DAKAR, "Region")
    employees = distinct_vals(T_DAKAR, "Employee_Name")

    div_sel = st.multiselect("Division", divisions)
    amo_sel = st.multiselect("AMO / Sales Office", amos)
    depo_sel = st.multiselect("Depo", depos)
    channel_sel = st.multiselect("Channel", channels)
    reg_sel = st.multiselect("Region", regions)
    emp_sel = st.multiselect("Employee", employees)

    saturday_weight = st.number_input("Saturday Weight", 0.0, 1.0, 0.5, 0.1)
    show_aggregation = st.checkbox("Show Aggregation (Division / AMO / Region / Depo + ranks)", True)

    apply_filters = st.button("🔄 Apply Filters", type="primary")

# ============================ Data Load ============================
if apply_filters or 'data_loaded' not in st.session_state:
    with st.spinner("Loading performance data..."):
        params = {"d1_str": d1.strftime("%Y-%m-%d"), "d2_str": d2.strftime("%Y-%m-%d")}
        cond = ["c.Sales_Date BETWEEN :d1_str AND :d2_str"]

        def add_in(col, vals, key):
            if vals:
                toks = []
                for i, v in enumerate(vals):
                    k = f"{key}{i}"
                    params[k] = v
                    toks.append(f":{k}")
                cond.append(f"{col} IN ({','.join(toks)})")

        add_in("d.Division",      div_sel, "div")
        add_in("d.AMO",           amo_sel, "amo")
        add_in("d.Depo",          depo_sel, "depo")
        add_in("d.Channel",       channel_sel, "ch")
        add_in("d.Region",        reg_sel, "reg")
        add_in("d.Employee_Name", emp_sel, "emp")
        where_clause = " AND ".join(cond)

        # ---- MAIN CMEC×DAKAR daily query (includes Tetap_Outlet) ----
        q_sales = f"""
            SELECT
                d.Division, d.AMO, d.Depo, d.Region,
                d.Employee_No_ AS Employee_ID, d.Employee_Name, d.Channel,
                CAST(c.Sales_Date AS DATE) AS Sales_Date,
                SUM(c.Tetap_outlet) AS Tetap_Outlet,
                SUM(c.CM_Tetap_1_) AS CM_Tetap,
                SUM(c.CM_Dummy_3_) AS CM_Dummy,
                SUM(c.EC_Tetap_2_) AS EC_Tetap,
                SUM(c.EC_Dummy_4_) AS EC_Dummy,
                SUM(c.Sales_Qty_)  AS Total_Volume,
                SUM(c.Sales_Price) AS Total_Sales
            FROM {T_CMEC} c
            INNER JOIN {T_DAKAR} d ON c.Employee_ID = d.Employee_No_
            WHERE {where_clause}
            GROUP BY d.Division, d.AMO, d.Depo, d.Region, d.Employee_No_, d.Employee_Name, d.Channel, CAST(c.Sales_Date AS DATE)
        """

        # ---- MNT pull for RO (normalized customer) ----
        mnt_cols = set(get_columns(T_MNT))
        has_ws = "Ws_Name" in mnt_cols
        q_mnt = f"""
            SELECT
                m.Region,
                m.AMO,
                {'m.Ws_Name,' if has_ws else ''}
                RIGHT(REPLICATE('0',8) + CAST(TRY_CONVERT(BIGINT, m.Customer) AS VARCHAR(32)), 8) AS Customer_Norm
            FROM {T_MNT} m
        """

        try:
            df_raw = pd.read_sql(text(q_sales), engine, params=params)
            if df_raw.empty:
                st.warning("No data found for the selected filters.")
                st.stop()

            df_mnt = pd.read_sql(text(q_mnt), engine)

            # ================= Working_Days (Hari Kerja) from actual CMEC rows =================
            df_raw["Sales_Date"] = pd.to_datetime(df_raw["Sales_Date"])
            df_raw["weekday"] = df_raw["Sales_Date"].dt.weekday  # Mon=0..Sun=6
            df_raw["wd_weight"] = np.where(
                df_raw["weekday"] == 6, 0.0,  # Sunday = 0
                np.where(df_raw["weekday"] == 5, float(saturday_weight), 1.0)  # Saturday weight, else 1
            )
            # Unique day per SR to avoid double counting if multiple rows same day
            wd_emp = (
                df_raw[["Employee_ID", "Sales_Date", "wd_weight"]]
                .drop_duplicates(subset=["Employee_ID", "Sales_Date"])
                .groupby("Employee_ID", as_index=False)["wd_weight"].sum()
                .rename(columns={"wd_weight": "Working_Days"})
            )

            # ================= SR totals in period =================
            df_emp = df_raw.groupby(
                ['Division','AMO','Depo','Region','Employee_ID','Employee_Name','Channel'],
                as_index=False
            ).agg({
                'Tetap_Outlet':'sum',
                'CM_Tetap':'sum','CM_Dummy':'sum',
                'EC_Tetap':'sum','EC_Dummy':'sum',
                'Total_Volume':'sum','Total_Sales':'sum'
            })
            df_emp = df_emp.merge(wd_emp, on="Employee_ID", how="left")
            df_emp["Working_Days"] = df_emp["Working_Days"].fillna(0.0)

            # ================= Robust RO =================
            # Normalize keys on BOTH sides
            for col in ["Region","AMO","Depo"]:
                if col in df_emp.columns:
                    df_emp[col] = df_emp[col].astype(str).str.strip().str.upper()
            for col in ["Region","AMO","Ws_Name"]:
                if col in df_mnt.columns:
                    df_mnt[col] = df_mnt[col].astype(str).str.strip().str.upper()

            ro_by_amo_ws = None
            if has_ws:
                ro_by_amo_ws = (
                    df_mnt.groupby(["AMO","Ws_Name"], as_index=False)["Customer_Norm"]
                          .nunique()
                          .rename(columns={"Customer_Norm":"RO_ws"})
                )
            ro_by_amo = (
                df_mnt.groupby(["AMO"], as_index=False)["Customer_Norm"]
                      .nunique()
                      .rename(columns={"Customer_Norm":"RO_amo"})
            )

            df_emp["RO"] = np.nan
            # 1) AMO+Depo ↔ AMO+Ws_Name
            if ro_by_amo_ws is not None:
                df_emp = df_emp.merge(
                    ro_by_amo_ws, left_on=["AMO","Depo"], right_on=["AMO","Ws_Name"], how="left"
                )
                if "RO_ws" in df_emp.columns:
                    df_emp["RO"] = df_emp["RO"].fillna(df_emp["RO_ws"])
                df_emp.drop(columns=[c for c in ["Ws_Name","RO_ws"] if c in df_emp.columns], inplace=True, errors="ignore")
            # 2) Fallback AMO
            df_emp = df_emp.merge(ro_by_amo, on="AMO", how="left")
            if "RO_amo" in df_emp.columns:
                df_emp["RO"] = df_emp["RO"].fillna(df_emp["RO_amo"])
            df_emp.drop(columns=[c for c in ["RO_amo"] if c in df_emp.columns], inplace=True, errors="ignore")
            df_emp["RO"] = df_emp["RO"].fillna(0).astype(int)

            # ================= Derived metrics =================
            df_emp['CM_Total'] = df_emp['CM_Tetap'] + df_emp['CM_Dummy']
            df_emp['EC_Total'] = df_emp['EC_Tetap'] + df_emp['EC_Dummy']

            # % vs Tetap_Outlet (consistent names)
            df_emp['CM Tetap %'] = pct_series(df_emp['CM_Tetap'], df_emp['Tetap_Outlet'])
            df_emp['EC Tetap %'] = pct_series(df_emp['EC_Tetap'], df_emp['Tetap_Outlet'])
            df_emp['CM Dummy %'] = pct_series(df_emp['CM_Dummy'], df_emp['Tetap_Outlet'])
            df_emp['EC Dummy %'] = pct_series(df_emp['EC_Dummy'], df_emp['Tetap_Outlet'])
            df_emp['EC Total %'] = pct_series(df_emp['EC_Total'], df_emp['Tetap_Outlet'])

            # Productivity per SR using actual Working_Days
            df_emp['Productivity'] = np.where(df_emp['Working_Days'] > 0,
                                              (df_emp['Total_Volume'] / df_emp['Working_Days']).round(2), 0.0)

            st.session_state['performance_data'] = df_emp.copy()
            st.session_state['saturday_weight'] = saturday_weight
            st.session_state['data_loaded'] = True

        except Exception as e:
            st.error(f"Error executing query: {e}")
            st.stop()

# ============================ Render ============================
if 'performance_data' in st.session_state:
    df_display = st.session_state['performance_data'].copy()

    # ---------- Employee table ----------
    st.subheader("Individual Employee Performance")
    st.dataframe(
        df_display.sort_values('Productivity', ascending=False),
        use_container_width=True,
        column_config={
            "Working_Days": st.column_config.NumberColumn("Working Days (actual)", format="%.1f"),
            "Tetap_Outlet": st.column_config.NumberColumn("Tetap Outlet", format="%.0f"),
            "RO": st.column_config.NumberColumn("RO (Distinct Cust)", format="%.0f"),
            "CM Tetap %": st.column_config.NumberColumn("CM Tetap %", format="%.1f%%"),
            "EC Tetap %": st.column_config.NumberColumn("EC Tetap %", format="%.1f%%"),
            "CM Dummy %": st.column_config.NumberColumn("CM Dummy %", format="%.1f%%"),
            "EC Dummy %": st.column_config.NumberColumn("EC Dummy %", format="%.1f%%"),
            "EC Total %": st.column_config.NumberColumn("EC Total %", format="%.1f%%"),
            "Total_Sales": st.column_config.NumberColumn("Total Sales", format="%.0f"),
            "Productivity": st.column_config.NumberColumn("Productivity (Vol/WD)", format="%.2f"),
        }
    )

    # ---------- Aggregations (incl. Depo & Ranks) ----------
    if show_aggregation:
        st.subheader("Performance Aggregation")

        def make_agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
            agg = df.groupby(group_cols, as_index=False).agg({
                "Tetap_Outlet":"sum",
                "CM_Tetap":"sum","CM_Dummy":"sum","CM_Total":"sum",
                "EC_Tetap":"sum","EC_Dummy":"sum","EC_Total":"sum",
                "Total_Volume":"sum","Total_Sales":"sum",
                "RO":"sum",
                "Working_Days":"mean",      # avg working days across SRs
                "Productivity":"mean"       # avg productivity across SRs
            })
            agg.rename(columns={"Working_Days":"Avg Working_Days",
                                "Productivity":"Avg Productivity"}, inplace=True)
            agg["CM Tetap %"] = pct_series(agg["CM_Tetap"], agg["Tetap_Outlet"])
            agg["EC Tetap %"] = pct_series(agg["EC_Tetap"], agg["Tetap_Outlet"])
            agg["CM Dummy %"] = pct_series(agg["CM_Dummy"], agg["Tetap_Outlet"])
            agg["EC Dummy %"] = pct_series(agg["EC_Dummy"], agg["Tetap_Outlet"])
            agg["EC Total %"] = pct_series(agg["EC_Total"], agg["Tetap_Outlet"])
            return agg

        div_agg  = make_agg(df_display, ["Division"])
        amo_agg  = make_agg(df_display, ["AMO"])
        reg_agg  = make_agg(df_display, ["Region"])
        depo_agg = make_agg(df_display, ["Division","AMO","Depo"])

        # Ranks inside Division–AMO
        depo_agg["Rank CM Tetap %"]   = depo_agg.groupby(["Division","AMO"])["CM Tetap %"].rank(method="dense", ascending=False).astype(int)
        depo_agg["Rank EC Tetap %"]   = depo_agg.groupby(["Division","AMO"])["EC Tetap %"].rank(method="dense", ascending=False).astype(int)
        depo_agg["Rank Productivity"] = depo_agg.groupby(["Division","AMO"])["Avg Productivity"].rank(method="dense", ascending=False).astype(int)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**By Division**")
            st.dataframe(div_agg.sort_values('Avg Productivity', ascending=False), use_container_width=True)
        with c2:
            st.write("**By AMO**")
            st.dataframe(amo_agg.sort_values('Avg Productivity', ascending=False), use_container_width=True)
        with c3:
            st.write("**By Region**")
            st.dataframe(reg_agg.sort_values('Avg Productivity', ascending=False), use_container_width=True)

        st.write("**By Depo (Division → AMO → Depo) + Ranks**")
        st.dataframe(
            depo_agg.sort_values(['Division','AMO','Rank Productivity','Rank CM Tetap %','Rank EC Tetap %']),
            use_container_width=True,
            column_config={
                "Tetap_Outlet": st.column_config.NumberColumn("Tetap Outlet", format="%.0f"),
                "RO": st.column_config.NumberColumn("RO (Distinct Cust)", format="%.0f"),
                "Avg Working_Days": st.column_config.NumberColumn("Avg Working Days", format="%.1f"),
                "Avg Productivity": st.column_config.NumberColumn("Avg Productivity", format="%.2f"),
                "CM Tetap %": st.column_config.NumberColumn("CM Tetap %", format="%.1f%%"),
                "EC Tetap %": st.column_config.NumberColumn("EC Tetap %", format="%.1f%%"),
                "CM Dummy %": st.column_config.NumberColumn("CM Dummy %", format="%.1f%%"),
                "EC Dummy %": st.column_config.NumberColumn("EC Dummy %", format="%.1f%%"),
                "EC Total %": st.column_config.NumberColumn("EC Total %", format="%.1f%%"),
            }
        )

    # ---------- Downloads ----------
    st.subheader("Download Data")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Individual Performance (CSV)",
            df_display.to_csv(index=False).encode('utf-8'),
            file_name=f"cm_ec_performance_{d1}_{d2}.csv",
            mime="text/csv"
        )
    with c2:
        pack = BytesIO()
        with pd.ExcelWriter(pack, engine='openpyxl') as xw:
            df_display.to_excel(xw, sheet_name='Employees', index=False)
            if show_aggregation:
                div_agg.to_excel(xw, sheet_name='Div_Agg', index=False)
                amo_agg.to_excel(xw, sheet_name='AMO_Agg', index=False)
                reg_agg.to_excel(xw, sheet_name='Reg_Agg', index=False)
                depo_agg.to_excel(xw, sheet_name='Depo_Agg', index=False)
        st.download_button(
            "Download Aggregations (Excel)",
            data=pack.getvalue(),
            file_name=f"aggregations_{d1}_{d2}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ---------- Overall KPIs ----------
    st.subheader("Overall Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Employees", len(df_display))
        st.metric("Avg Working Days", f"{df_display['Working_Days'].mean():.1f}")
    with c2:
        st.metric("Total Volume", f"{df_display['Total_Volume'].sum():,.0f}")
        st.metric("Total Sales", f"{df_display['Total_Sales'].sum():,.0f}")
    with c3:
        st.metric("Total Tetap Outlet", f"{df_display['Tetap_Outlet'].sum():,.0f}")
        st.metric("Distinct RO (MNT)", f"{df_display['RO'].sum():,.0f}")
    with c4:
        st.metric("Avg Productivity", f"{df_display['Productivity'].mean():.2f}")

    # ---------- Optional Channel overview ----------
    st.subheader("Channel Performance Overview")
    ch = df_display.groupby('Channel', as_index=False).agg({
        'CM_Total':'sum','EC_Total':'sum','Total_Volume':'sum','Total_Sales':'sum',
        'Working_Days':'mean','Productivity':'mean'
    }).rename(columns={
        'CM_Total':'Total_CM','EC_Total':'Total_EC',
        'Total_Volume':'Volume','Total_Sales':'Sales',
        'Working_Days':'Avg_Working_Days','Productivity':'Avg_Productivity'
    })
    fig = px.bar(ch, x='Channel', y=['Total_CM','Total_EC'], barmode='group', title="CM vs EC by Channel")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ch, use_container_width=True)
