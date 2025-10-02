# pages/02_Weekly_Comparison.py
import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import plotly.express as px

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

# ============================ Page config ============================
st.set_page_config(page_title="Weekly Comparison & Analysis", layout="wide")
st.title("📆 Weekly Comparison & Analysis")
st.caption("Independent page: filters & data are queried directly from SQL Server.")

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

# Table names from secrets (fallbacks provided)
T_CMEC  = st.secrets.get("tables", {}).get("cmec",  "dbo.cmec")
T_DAKAR = st.secrets.get("tables", {}).get("dakar", "dbo.dakar")

# ============================ Sidebar Filters ============================
@st.cache_data(ttl=900, show_spinner=False)
def distinct_vals(table: str, col: str):
    try:
        q = text(f"SELECT DISTINCT {col} AS v FROM {table} WHERE {col} IS NOT NULL ORDER BY v")
        return pd.read_sql(q, engine)["v"].astype(str).tolist()
    except Exception:
        return []

with st.sidebar:
    st.subheader("🔎 Filters")

    # Dates
    today = dt.date.today()
    d1, d2 = st.date_input("Sales Date Range", (dt.date(today.year, 1, 1), today))

    # Dimensions (from DAKAR)
    div_sel  = st.multiselect("Division", distinct_vals(T_DAKAR, "Division"), default=[])
    amo_sel  = st.multiselect("AMO / Sales Office", distinct_vals(T_DAKAR, "AMO"), default=[])
    depo_sel = st.multiselect("Depo", distinct_vals(T_DAKAR, "Depo"), default=[])
    ch_sel   = st.multiselect("Channel", distinct_vals(T_DAKAR, "Channel"), default=[])
    reg_sel  = st.multiselect("Region", distinct_vals(T_DAKAR, "Region"), default=[])
    emp_sel  = st.multiselect("Employee", distinct_vals(T_DAKAR, "Employee_Name"), default=[])

    saturday_weight = st.number_input("Saturday Weight", 0.0, 1.0, 0.5, 0.1)
    st.markdown("---")

    # Time + value mode + charts
    grain = st.radio("Time Grain", ["Week", "Month", "Day"], index=0, horizontal=True)
    value_mode = st.radio("Value Mode", ["Sum", "Daily Average"], index=0, horizontal=True)
    periods_back = st.slider("How many periods to display", 3, 20, 8)

    # Retail baseline rule (for Week grain only)
    retail_week_rule = st.selectbox(
        "Retail (Week) Baseline",
        ["Same parity (even↔even / odd↔odd)", "Immediate previous week"],
        index=0,
        help="Only applies when Channel is Retail and Time Grain is Week"
    )

    st.markdown("---")
    cm_metric = st.selectbox("CM metric", ["CM_Tetap", "CM_Dummy", "CM_Total"], index=2,
                             help="CM_Total = CM_Tetap + CM_Dummy")
    ec_metric = st.selectbox("EC metric", ["EC_Tetap", "EC_Dummy", "EC_Total"], index=2,
                             help="EC_Total = EC_Tetap + EC_Dummy")

    st.markdown("---")
    show_charts = st.checkbox("Show charts", value=True)
    chart_type = st.selectbox("Chart type", ["Line", "Bar"], index=0)

    apply_btn = st.button("🔄 Apply / Refresh", type="primary")

# ============================ Query & Prep ============================
def build_where(params_in):
    where = ["c.Sales_Date BETWEEN :d1_str AND :d2_str"]
    params = {"d1_str": d1.strftime("%Y-%m-%d"), "d2_str": d2.strftime("%Y-%m-%d")}

    def add_filter(col, sel, prefix):
        if sel:
            toks = []
            for i, v in enumerate(sel):
                key = f"{prefix}{i}"
                params[key] = v
                toks.append(f":{key}")
            where.append(f"{col} IN ({','.join(toks)})")

    add_filter("d.Division",      params_in.get("Division"),      "div")
    add_filter("d.AMO",           params_in.get("AMO"),           "amo")
    add_filter("d.Depo",          params_in.get("Depo"),          "depo")
    add_filter("d.Channel",       params_in.get("Channel"),       "ch")
    add_filter("d.Region",        params_in.get("Region"),        "reg")
    add_filter("d.Employee_Name", params_in.get("Employee_Name"), "emp")
    return " AND ".join(where), params

def run_query():
    sels = {
        "Division": div_sel, "AMO": amo_sel, "Depo": depo_sel,
        "Channel": ch_sel, "Region": reg_sel, "Employee_Name": emp_sel
    }
    where_clause, params = build_where(sels)
    q = f"""
        SELECT
            d.Division, d.AMO, d.Depo, d.Region, d.Employee_Name, d.Channel,
            CAST(c.Sales_Date AS DATE) AS Sales_Date,
            c.CM_Tetap_1_   AS CM_Tetap,
            c.CM_Dummy_3_   AS CM_Dummy,
            c.EC_Tetap_2_   AS EC_Tetap,
            c.EC_Dummy_4_   AS EC_Dummy,
            c.Sales_Qty_    AS Total_Volume,
            c.Sales_Price   AS Total_Sales
        FROM {T_CMEC} c
        INNER JOIN {T_DAKAR} d ON c.Employee_ID = d.Employee_No_
        WHERE {where_clause}
    """
    return pd.read_sql(text(q), engine, params=params)

if apply_btn or "weekly_raw" not in st.session_state:
    with st.spinner("Querying data..."):
        try:
            df_raw = run_query()
            if df_raw.empty:
                st.warning("No data found for the selected filters.")
                st.stop()
            st.session_state["weekly_raw"] = df_raw
            st.session_state["saturday_weight"] = saturday_weight
        except Exception as e:
            st.error(f"Query error: {e}")
            st.stop()

df_raw = st.session_state["weekly_raw"].copy()
saturday_weight = st.session_state.get("saturday_weight", saturday_weight)

# ============================ Helpers ============================
def ensure_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sales_Date"] = pd.to_datetime(df["Sales_Date"])
    iso = df["Sales_Date"].dt.isocalendar()
    df["Year"]  = iso.year.astype(int)
    df["Week"]  = iso.week.astype(int)
    df["Month"] = df["Sales_Date"].dt.to_period("M").astype(str)
    return df

def add_working_weight(df: pd.DataFrame, sat_w: float) -> pd.DataFrame:
    df = df.copy()
    wd = df["Sales_Date"].dt.weekday  # 0..6 (Sun=6)
    df["Working_Days"] = wd.map({6: 0}).fillna(1.0)
    df.loc[wd == 5, "Working_Days"] = float(sat_w)
    return df

def aggregate_by_grain(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    df = ensure_time_parts(df)
    df = add_working_weight(df, saturday_weight)

    if grain == "Week":
        df["period_key"]   = df["Year"] * 100 + df["Week"]        # YYYYWW
        df["period_label"] = df["Week"].astype(int)
        group_cols = ["Channel", "Year", "Week", "period_key", "period_label"]
        # Day baseline helpers not needed here
    elif grain == "Month":
        df["period_key"]   = df["Sales_Date"].dt.year * 100 + df["Sales_Date"].dt.month  # YYYYMM
        df["period_label"] = df["Sales_Date"].dt.strftime("%Y-%m")
        group_cols = ["Channel", "period_key", "period_label"]
    else:  # Day
        df["period_key"]   = df["Sales_Date"].dt.strftime("%Y%m%d").astype(int)          # YYYYMMDD
        df["period_label"] = df["Sales_Date"].dt.strftime("%Y-%m-%d")
        df["period_date"]  = df["Sales_Date"].dt.date
        group_cols = ["Channel", "period_key", "period_label", "period_date"]

    agg = (
        df.groupby(group_cols, as_index=False)
          .agg({
              "CM_Tetap":"sum", "CM_Dummy":"sum",
              "EC_Tetap":"sum", "EC_Dummy":"sum",
              "Total_Volume":"sum", "Total_Sales":"sum",
              "Working_Days":"sum"
          })
    )
    agg["CM_Total"] = agg["CM_Tetap"] + agg["CM_Dummy"]
    agg["EC_Total"] = agg["EC_Tetap"] + agg["EC_Dummy"]
    return agg

def label_for_key_short(k: int, grain: str) -> str:
    if grain == "Week":
        return f"W{int(k % 100):02d}"
    if grain == "Month":
        y, m = divmod(int(k), 100)
        return f"{y}-{m:02d}"
    s = str(int(k))
    return f"{s[2:4]}/{s[4:6]}"  # YY/MM short

def label_for_key_long(k: int, grain: str) -> str:
    if grain == "Week":
        return f"{int(k // 100)}-W{int(k % 100):02d}"
    if grain == "Month":
        y, m = divmod(int(k), 100)
        return f"{y}-{m:02d}"
    s = str(int(k))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"

def color_change(val: str) -> str:
    if val == "-": return ""
    try:
        x = float(val.replace("%","").replace("+",""))
        if x > 0: return "color: green; font-weight: 600"
        if x < 0: return "color: red; font-weight: 600"
    except: pass
    return ""

def previous_key_for_channel(
    channel: str,
    current_key: int,
    keys_in_window: list[int],
    grain: str,
    retail_week_rule: str,
    key_to_date: dict[int, dt.date] | None = None
):
    """
    - Week & Retail:
        * 'Same parity...' -> closest previous with same week parity (even/odd).
        * 'Immediate previous week' -> immediate previous key.
      Others on Week -> immediate previous.
    - Month -> immediate previous.
    - Day -> key 14 days earlier (same weekday two weeks ago).
    """
    if current_key not in keys_in_window:
        return None
    idx = keys_in_window.index(current_key)
    if idx == 0:
        return None

    if grain == "Week":
        if channel.lower() == "retail" and retail_week_rule.startswith("Same parity"):
            want = (current_key % 100) % 2
            for j in range(idx - 1, -1, -1):
                if (keys_in_window[j] % 100) % 2 == want:
                    return keys_in_window[j]
            return None
        else:
            return keys_in_window[idx - 1]

    if grain == "Day":
        if key_to_date is None or current_key not in key_to_date:
            return None
        curr_date = key_to_date[current_key]
        target = curr_date - timedelta(days=14)  # two weeks ago, same weekday
        target_key = int(target.strftime("%Y%m%d"))
        return target_key if target_key in keys_in_window else None

    # Month
    return keys_in_window[idx - 1]

def make_chart(df_series: pd.DataFrame, metric_label: str, grain: str, chart_type: str):
    # df_series: columns -> Channel, period_key, period_label_long, value
    df_series = df_series.copy()
    df_series["period_order"] = df_series["period_key"].rank(method="dense").astype(int)

    if chart_type == "Bar":
        fig = px.bar(
            df_series,
            x="period_label_long",
            y="value",
            color="Channel",
            barmode="group",
            title=f"{metric_label} by {grain}",
        )
    else:
        fig = px.line(
            df_series,
            x="period_label_long",
            y="value",
            color="Channel",
            markers=True,
            title=f"{metric_label} Trend by {grain}",
        )
    fig.update_layout(xaxis_title=grain, yaxis_title=metric_label, height=420, legend_title_text="Channel")
    return fig

def build_table_and_series(
    agg: pd.DataFrame,
    metric: str,
    metric_label: str,
    grain: str,
    periods_back: int,
    value_mode: str,
    retail_week_rule: str
):
    as_daily_avg = (value_mode == "Daily Average")

    # Keys & (for Day) map key->date
    keys_sorted = sorted(agg["period_key"].unique())
    if not keys_sorted:
        return None, None

    # Per-section "End at" picker (shares state with table+chart)
    cols_top = st.columns([1.2, 3, 1, 1, 1])
    with cols_top[0]:
        end_options = ["Current"] + [int(k) for k in keys_sorted]
        labels = ["Current"] + [label_for_key_short(k, grain) for k in keys_sorted]
        end_choice_label = st.selectbox("End at", labels, key=f"end_{metric}_{grain}")
    end_choice = end_options[labels.index(end_choice_label)]
    end_key = keys_sorted[-1] if end_choice == "Current" else int(end_choice)

    window = [k for k in keys_sorted if k <= end_key]
    window = window[-periods_back:] if len(window) >= periods_back else window

    dfw = agg[agg["period_key"].isin(window)].copy()
    if dfw.empty:
        return None, None

    # Daily Average option (divide by Working_Days before aggregating)
    if as_daily_avg and metric != "Working_Days":
        dfw[metric] = np.where(dfw["Working_Days"] > 0, dfw[metric] / dfw["Working_Days"], 0.0)

    # For Day baseline we need actual date
    key_to_date = None
    if grain == "Day":
        if "period_date" in dfw.columns:
            key_to_date = dfw[["period_key", "period_date"]].drop_duplicates().set_index("period_key")["period_date"].to_dict()

    # Build pivot table (Channel x period_key)
    aggfunc = np.sum if value_mode == "Sum" else np.mean
    pv = dfw.pivot_table(index="Channel", columns="period_key", values=metric, aggfunc=aggfunc).fillna(0.0)
    pv = pv.reindex(columns=sorted(pv.columns))

    current_num = pv.columns[-1]
    keys_in_window = list(pv.columns)

    # Baseline (per channel)
    comp = {
        ch: previous_key_for_channel(ch, current_num, keys_in_window, grain, retail_week_rule, key_to_date)
        for ch in pv.index
    }

    # % vs baseline
    vs_prev = []
    for ch in pv.index:
        base = comp[ch]
        if base is None or pv.loc[ch, base] == 0:
            vs_prev.append(0.0)
        else:
            vs_prev.append(round((pv.loc[ch, current_num] - pv.loc[ch, base]) / pv.loc[ch, base] * 100, 1))
    vs_prev = pd.Series(vs_prev, index=pv.index)

    # % vs 4-back (simple numeric)
    if len(pv.columns) >= 5:
        four_back = pv.columns[-5]
        denom = pv[four_back].replace(0, np.nan)
        vs_l4 = ((pv[current_num] - pv[four_back]) / denom * 100).fillna(0).round(1)
    else:
        vs_l4 = pd.Series(0.0, index=pv.index)

    # Rename last numeric to 'Current'
    cols_num = list(pv.columns)
    cols_num[-1] = "Current"
    pv.columns = cols_num

    # Baseline numeric col
    baseline_vals = []
    for ch in pv.index:
        bk = comp[ch]
        if bk is None:
            baseline_vals.append(0.0)
        else:
            tmp = pv.rename(columns={"Current": current_num})
            baseline_vals.append(float(tmp.loc[ch, bk]))
    pv["Baseline"] = baseline_vals

    # Assemble display
    hist = [c for c in pv.columns if isinstance(c, (int, np.integer))]
    disp = pv[hist + ["Baseline", "Current"]].copy()

    # Formatting
    if metric in ["Total_Volume", "Total_Sales"]:
        for c in disp.columns:
            disp[c] = disp[c].apply(lambda x: f"{x:,.0f}" if x != 0 else "-")
    else:
        for c in disp.columns:
            disp[c] = disp[c].apply(lambda x: f"{x:.1f}" if x != 0 else "-")

    disp["vs LW"] = vs_prev.apply(lambda x: f"{x:+.1f}%" if x != 0 else "-")
    disp["vs L4"] = vs_l4.apply(lambda x: f"{x:+.1f}%" if x != 0 else "-")

    disp.rename(columns={c: label_for_key_short(c, grain) for c in hist}, inplace=True)

    styled = (
        disp.style
            .applymap(color_change, subset=["vs LW", "vs L4"])
            .set_table_styles([
                {"selector":"th","props":[("background-color","#f0f2f6"),("color","black"),("font-weight","bold")]},
                {"selector":"td","props":[("text-align","center")]},
            ])
    )

    # -------- Build tidy series for chart (same window & values) --------
    # Rebuild series from dfw using same aggregation logic
    series = (
        dfw.groupby(["Channel", "period_key"], as_index=False)[metric]
           .agg(aggfunc)
           .rename(columns={metric: "value"})
    )
    series = series[series["period_key"].isin(window)]
    series["period_label_long"] = series["period_key"].apply(lambda k: label_for_key_long(k, grain))

    return styled, series

# ============================ Build Aggregates ============================
agg = aggregate_by_grain(df_raw, grain)

# ============================ Render CM ============================
st.subheader("CM Comparison")
cm_table, cm_series = build_table_and_series(agg, cm_metric, "CM", grain, periods_back, value_mode, retail_week_rule)
if cm_table is not None:
    st.dataframe(cm_table, use_container_width=True)
    if show_charts and cm_series is not None and not cm_series.empty:
        fig = make_chart(cm_series, metric_label=cm_metric if value_mode=="Sum" else f"{cm_metric} (Daily Avg)",
                         grain=grain, chart_type=chart_type)
        st.plotly_chart(fig, use_container_width=True)
st.caption("Retail (Week): choose parity vs immediate previous in the sidebar. Day: baseline is two weeks ago same weekday.")

# ============================ Render EC ============================
st.subheader("EC Comparison")
ec_table, ec_series = build_table_and_series(agg, ec_metric, "EC", grain, periods_back, value_mode, retail_week_rule)
if ec_table is not None:
    st.dataframe(ec_table, use_container_width=True)
    if show_charts and ec_series is not None and not ec_series.empty:
        fig = make_chart(ec_series, metric_label=ec_metric if value_mode=="Sum" else f"{ec_metric} (Daily Avg)",
                         grain=grain, chart_type=chart_type)
        st.plotly_chart(fig, use_container_width=True)

# ============================ Render Sales ============================
st.subheader("Total Sales")
sales_table, sales_series = build_table_and_series(agg, "Total_Sales", "Total Sales", grain, periods_back, value_mode, retail_week_rule)
if sales_table is not None:
    st.dataframe(sales_table, use_container_width=True)
    if show_charts and sales_series is not None and not sales_series.empty:
        fig = make_chart(sales_series, metric_label="Total_Sales" if value_mode=="Sum" else "Total_Sales (Daily Avg)",
                         grain=grain, chart_type=chart_type)
        st.plotly_chart(fig, use_container_width=True)

# ============================ Render Productivity ============================
st.subheader("Productivity")
agg_prod = agg.copy()
agg_prod["Productivity"] = np.where(
    agg_prod["Working_Days"] > 0,
    agg_prod["Total_Volume"] / agg_prod["Working_Days"],
    0.0
)
prod_table, prod_series = build_table_and_series(agg_prod, "Productivity", "Productivity", grain, periods_back, "Sum", retail_week_rule)
# note: Productivity already normalized per day; treat as "Sum" to aggregate across rows.
if prod_table is not None:
    st.dataframe(prod_table, use_container_width=True)
    if show_charts and prod_series is not None and not prod_series.empty:
        fig = make_chart(prod_series, metric_label="Productivity", grain=grain, chart_type=chart_type)
        st.plotly_chart(fig, use_container_width=True)
