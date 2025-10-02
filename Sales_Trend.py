# app_smsdv_timegrain_units.py
# Sales Qty dashboard for dbo.smsdv_desc (or dbo.smsdv) with units + time grains

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



st.set_page_config(page_title="Sales Qty Dashboard", layout="wide")
st.title("📊 Sales Trend Dashboard")

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
T_DESC = st.secrets.get("tables", {}).get("smsdv_desc", "dbo.smsdv_desc")
T_MAIN = st.secrets.get("tables", {}).get("smsdv",      "dbo.smsdv")

@st.cache_data(ttl=1800, show_spinner=False)
def table_columns(table: str) -> list[str]:
    try:
        q = sa_text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = PARSENAME(:t,1)
              AND (TABLE_SCHEMA = PARSENAME(:t,2) OR :t NOT LIKE '%.%')
            ORDER BY ORDINAL_POSITION
        """)
        return pd.read_sql(q, engine, params={"t": table})["COLUMN_NAME"].str.strip().tolist()
    except Exception:
        return []

ALL_DESC = table_columns(T_DESC)
ALL_MAIN = table_columns(T_MAIN)

def pick_col(candidates, available):
    al = {c.lower(): c for c in available}
    for c in candidates:
        if c and c.lower() in al:
            return al[c.lower()]
    # loose match (ignore space/underscore)
    for c in candidates:
        if not c: continue
        want = c.lower().replace(" ", "").replace("_", "")
        for a in available:
            if a.lower().replace(" ", "").replace("_", "") == want:
                return a
    return None

def choose_active_table():
    date_desc = pick_col(["converted_date","Sales_Date","Visit_Date","Tanggal","Date"], ALL_DESC)
    qty_desc  = pick_col(["Sales_Qty_","Qty","Quantity"], ALL_DESC)
    if ALL_DESC and date_desc and qty_desc:
        return T_DESC, ALL_DESC
    return T_MAIN, ALL_MAIN

TABLE, ALL = choose_active_table()

DATE_COL   = pick_col(["converted_date","Visit_Date","Tanggal","Date"], ALL)
QTY_PACK   = pick_col(["Sales_Qty_","Qty","Quantity"], ALL)
PRICE_COL  = pick_col(["Sales_Price","Sales_Amount","Total_Sales"], ALL)
SKU_COL    = pick_col(["SKU","Product_Name","SKU_Name"], ALL)
SKU_CODE   = pick_col(["Prod_Code","Product_Code","SKU_Code"], ALL)
REGION_COL = pick_col(["Region","Region_Code","Area"], ALL)
WS_COL     = pick_col(["Ws_Name","WH_Name","Warehouse","From_W_H","Depo","WsName"], ALL)
CH_HI      = pick_col(["Channel_High_","Channel_High"], ALL)
CH_MD      = pick_col(["Channel_Mid_","Channel_Mid"], ALL)
CH_LO      = pick_col(["Channel_Low_","Channel_Low","Channel"], ALL)

BOX_QTY_COL       = pick_col(["Box_Qty","Box","BoxQty"], ALL)
BOX_FACTOR_COL    = pick_col(["Pack_in_Box","PackInBox","pack_in_box","BoxPer","Box"], ALL)
STICKS_QTY_COL    = pick_col(["Sticks_Qty","Qty_Stick"], ALL)
STICKS_FACTOR_COL = pick_col(["Sticks","Sticks_per_Pack","Stick_in_Pack","stick_in_pack"], ALL)

if not DATE_COL or not QTY_PACK:
    st.error("Date or quantity column not found. Please adjust column mappings at the top.")
    st.stop()

@st.cache_data(ttl=900, show_spinner=False)
def get_date_bounds():
    q = sa_text(f"SELECT CAST(MIN({DATE_COL}) AS date) mn, CAST(MAX({DATE_COL}) AS date) mx FROM {TABLE}")
    r = pd.read_sql(q, engine)
    if r.empty or pd.isna(r.loc[0,"mn"]) or pd.isna(r.loc[0,"mx"]):
        today = dt.date.today()
        return today, today
    return r.loc[0,"mn"], r.loc[0,"mx"]

dmin, dmax = get_date_bounds()

# ---------- Distinct helper ----------
@st.cache_data(ttl=900, show_spinner=False)
def distinct_vals(table: str, date_col: str, col: str, start, end):
    if not col:
        return []
    sql = sa_text(
        f"SELECT DISTINCT {col} AS v FROM {table} "
        f"WHERE CAST({date_col} AS date) BETWEEN :d1 AND :d2 AND {col} IS NOT NULL ORDER BY v"
    )
    try:
        return pd.read_sql(sql, engine, params={"d1": start, "d2": end})["v"].astype(str).tolist()
    except Exception:
        return []

# ---------- Filter bar ----------
with st.container():
    st.subheader("Filters")
    with st.form("filters"):
        r1c1, r1c2, r1c3, r1c4 = st.columns([2, 1.2, 1.2, 1.6])
        with r1c1:
            dr = st.date_input("Date range", (dmin, dmax))
            if isinstance(dr, (list, tuple)):
                d1, d2 = dr[0], dr[1]
            else:
                d1, d2 = dr, dr
            if d1 > d2: d1, d2 = d2, d1
        with r1c2:
            unit = st.radio("Unit", ["Pack","Box","Sticks"], index=0, horizontal=True)
        with r1c3:
            timegrain = st.radio("Time grain", ["Daily","Weekly","Monthly"], index=1, horizontal=True)
        with r1c4:
            ch_level = st.radio("Channel level", ["Low","Mid","High"], horizontal=True, index=0)
            CH_COL = {"Low": CH_LO, "Mid": CH_MD, "High": CH_HI}.get(ch_level, CH_LO)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            wss = distinct_vals(TABLE, DATE_COL, WS_COL, d1, d2) if WS_COL else []
            ws_sel  = st.multiselect("WS (Warehouse)", wss, default=[])
        with r2c2:
            channels = distinct_vals(TABLE, DATE_COL, CH_COL, d1, d2) if CH_COL else []
            ch_sel  = st.multiselect(f"Channel {ch_level}", channels, default=[])
        with r2c3:
            regions  = distinct_vals(TABLE, DATE_COL, REGION_COL, d1, d2) if REGION_COL else []
            reg_sel = st.multiselect("Region", regions, default=[])
        with r2c4:
            skus     = distinct_vals(TABLE, DATE_COL, SKU_COL, d1, d2) if SKU_COL else []
            sku_sel = st.multiselect("SKU (SKU)", skus, default=[])

        r3c1, r3c2 = st.columns([1,1])
        with r3c1:
            use_kategori = st.checkbox("Use kategori (brand from SKU)", value=True)
        with r3c2:
            max_periods = st.number_input("Max periods to plot", min_value=12, max_value=520, value=104, step=4)

        submitted = st.form_submit_button("Apply")

# ---------- WHERE & params ----------
def in_clause(col, values, params, prefix):
    if not col or not values:
        return None, params
    new_params = dict(params)
    ks = []
    for i, v in enumerate(values):
        k = f"{prefix}{i}"
        new_params[k] = v
        ks.append(f":{k}")
    return f"{col} IN ({', '.join(ks)})", new_params

conditions = []
params = {"d1": d1, "d2": d2}
conditions.append(f"CAST({DATE_COL} AS date) BETWEEN :d1 AND :d2")

cond, params = in_clause(WS_COL, ws_sel, params, "ws")
if cond: conditions.append(cond)
cond, params = in_clause(CH_COL, ch_sel, params, "ch")
if cond: conditions.append(cond)
cond, params = in_clause(REGION_COL, reg_sel, params, "rg")
if cond: conditions.append(cond)
cond, params = in_clause(SKU_COL, sku_sel, params, "sk")
if cond: conditions.append(cond)

WHERE_SQL = (" WHERE " + " AND ".join(conditions)) if conditions else ""

# ---------- SQL helpers ----------
def qty_expression_sql(unit: str) -> str:
    base = QTY_PACK
    if unit == "Pack":
        return base
    if unit == "Box":
        if BOX_QTY_COL:
            return f"CAST({BOX_QTY_COL} AS float)"
        if BOX_FACTOR_COL:
            return f"CAST(({base}) / NULLIF({BOX_FACTOR_COL}, 0) AS float)"
        return f"CAST({base} AS float)"
    if unit == "Sticks":
        if STICKS_QTY_COL:
            return f"CAST({STICKS_QTY_COL} AS float)"
        if STICKS_FACTOR_COL:
            return f"CAST(({base}) * NULLIF({STICKS_FACTOR_COL}, 0) AS float)"
        return f"CAST({base} AS float)"
    return f"CAST({base} AS float)"

def period_sql(grain: str) -> str:
    # base CTE exposes DATE_COL as alias 'd'
    if grain == "Daily":
        return "CONVERT(varchar(10), CAST(d AS date), 23)"  # yyyy-mm-dd
    if grain == "Weekly":
        # Monday-of-week, independent of @@DATEFIRST
        return (
            "CONVERT(varchar(10), "
            "DATEADD(day, - ((DATEPART(weekday, d) + @@DATEFIRST - 2) % 7), CAST(d AS date)), 23)"
        )
    # Monthly key: YYYY-MM
    return "LEFT(CONVERT(varchar(10), DATEFROMPARTS(YEAR(d), MONTH(d), 1), 23), 7)"

def base_cte(unit: str, include_region=True, include_sku=True, include_ws=True, include_channel=True):
    cols = [f"{DATE_COL} AS d", f"{qty_expression_sql(unit)} AS qty"]
    if include_region and REGION_COL: cols.append(f"{REGION_COL} AS region")
    if include_sku and SKU_COL:       cols.append(f"{SKU_COL} AS sku")
    if include_ws and WS_COL:         cols.append(f"{WS_COL} AS ws")
    if include_channel and CH_COL:    cols.append(f"{CH_COL} AS channel")
    select_cols = ", ".join(cols)
    return f"WITH base AS (SELECT {select_cols} FROM {TABLE}{WHERE_SQL})"

# ---------- Queries ----------
@st.cache_data(ttl=600, show_spinner=False)
def query_agg(unit: str, grain: str, max_periods: int, params: dict) -> pd.DataFrame:
    psql = period_sql(grain)
    sql = f"""
        {base_cte(unit, include_region=False, include_sku=False, include_ws=False, include_channel=False)}
        , series AS (
            SELECT {psql} AS period, SUM(CAST(qty AS float)) AS value
            FROM base
            GROUP BY {psql}
        )
        SELECT TOP ({int(max_periods)}) period, value
        FROM series
        ORDER BY period DESC
    """
    df = pd.read_sql(sa_text(sql), engine, params=params)
    return df.sort_values("period").reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def query_breakdown(unit: str, label: str, params: dict) -> pd.DataFrame:
    if label == "Region":
        sel = "region"
    elif label == "SKU":
        sel = "sku"
    elif label == "Kategori":
        # first token as brand
        sel = "UPPER(LEFT(sku, NULLIF(CHARINDEX(' ', sku + ' ')-1, -1)))"
    else:
        sel = "sku"

    sql = f"""
        {base_cte(unit, include_region=True, include_sku=True, include_ws=False, include_channel=False)}
        SELECT {sel} AS label, SUM(CAST(qty AS float)) AS qty
        FROM base
        GROUP BY {sel}
    """
    df = pd.read_sql(sa_text(sql), engine, params=params)
    if df.empty:
        df["share"] = []
        return df
    df = df.sort_values("qty", ascending=False)
    total = df["qty"].sum()
    df["share"] = (df["qty"] / total).astype(float)
    return df

@st.cache_data(ttl=600, show_spinner=False)
def query_channel_share(unit: str, params: dict) -> pd.DataFrame:
    if not CH_COL:
        return pd.DataFrame(columns=["label","qty","share"])
    sql = f"""
        {base_cte(unit, include_region=False, include_sku=False, include_ws=False, include_channel=True)}
        SELECT channel AS label, SUM(CAST(qty AS float)) AS qty
        FROM base
        GROUP BY channel
    """
    df = pd.read_sql(sa_text(sql), engine, params=params)
    if df.empty:
        df["share"] = []
        return df
    df = df.sort_values("qty", ascending=False)
    df["share"] = df["qty"] / df["qty"].sum()
    return df

@st.cache_data(ttl=600, show_spinner=False)
def query_breakdown_by_period(unit: str, label: str, grain: str, n_periods: int, params: dict) -> pd.DataFrame:
    if label == "SKU":
        sel = "sku"
    elif label == "Kategori":
        sel = "UPPER(LEFT(sku, NULLIF(CHARINDEX(' ', sku + ' ')-1, -1)))"
    else:
        sel = "region"
    psql = period_sql(grain)
    sql = f"""
        {base_cte(unit, include_region=True, include_sku=True, include_ws=False, include_channel=False)}
        , series AS (
            SELECT {psql} AS period, {sel} AS label, SUM(CAST(qty AS float)) AS qty
            FROM base
            GROUP BY {psql}, {sel}
        )
        , lastp AS (
            SELECT DISTINCT TOP ({int(n_periods)}) period FROM series ORDER BY period DESC
        )
        SELECT s.period, s.label, s.qty
        FROM series s
        JOIN lastp p ON p.period = s.period
    """
    return pd.read_sql(sa_text(sql), engine, params=params)

# ---------- Labels ----------
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _fmt_day(d: dt.date) -> str:
    return f"{d.day} {MONTH_ABBR[d.month-1]}"

def weekly_label_from_key(key: str) -> str:
    # key 'YYYY-MM-DD' Monday-of-week
    start = pd.to_datetime(key).date()
    end = start + dt.timedelta(days=6)
    week_no = start.isocalendar()[1]
    return f"W{week_no:02d} ({_fmt_day(start)}–{_fmt_day(end)} {end.year})"

def monthly_label_from_key(key: str) -> str:
    year, month = map(int, key.split("-"))
    first = dt.date(year, month, 1)
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    return f"{MONTH_ABBR[month-1]} {year} ({_fmt_day(first)}–{_fmt_day(last)} {year})"

def add_period_labels(agg: pd.DataFrame, grain: str) -> pd.DataFrame:
    if agg.empty:
        agg["label"] = []
        return agg
    if grain == "Weekly":
        agg["label"] = agg["period"].apply(weekly_label_from_key)
    elif grain == "Monthly":
        agg["label"] = agg["period"].apply(monthly_label_from_key)
    else:
        agg["label"] = pd.to_datetime(agg["period"]).dt.strftime("%d %b %Y")
    return agg

def pretty_label_from_key(key: str, grain: str) -> str:
    return weekly_label_from_key(key) if grain == "Weekly" else (
        monthly_label_from_key(key) if grain == "Monthly" else key
    )

# ---------- Run main series ----------
with st.spinner("Fetching aggregates…"):
    agg = query_agg(unit, timegrain, int(max_periods), params)
    agg = add_period_labels(agg, timegrain)

# ---------- KPIs + Trend ----------
left, right = st.columns([1,3])

with left:
    st.subheader("")
    total_qty = float(agg["value"].sum()) if not agg.empty else 0.0
    st.metric(f"Total {unit.lower()} (periods shown)", f"{total_qty:,.0f}")
    if len(agg) >= 2:
        last2 = agg.tail(2)["value"].tolist()
        diff  = last2[-1] - last2[-2]
        gname = timegrain[:-2].lower() if timegrain.endswith("ly") else timegrain.lower()
        st.metric(f"Last vs prev {gname}", f"{last2[-1]:,.0f}", f"{diff:+,.0f}")

with right:
    st.subheader(f"{timegrain} {unit} Trend")
    if agg.empty:
        st.info("No data for the current filters.")
    else:
        base = alt.Chart(agg).encode(
            x=alt.X("label:N", title=timegrain, axis=alt.Axis(labelAngle=0, labelLimit=2000, labelOverlap=False)),
            y=alt.Y("value:Q", title=f"Trend ({unit})")
        )
        line_layer = base.mark_line(point=True)
        label_layer = base.mark_text(align="left", dx=6, dy=-6).encode(
            text=alt.Text("value:Q", format=",.0f")
        )
        st.altair_chart((line_layer + label_layer).properties(height=380), use_container_width=True)

# ---------- Analysis tabs ----------
st.subheader("Sales Analysis")
tab_brand, tab_product, tab_channel, tab_region, tab_movers = st.tabs(
    ["Brand contribution", "Product contribution", "Channel mix", "Region contribution", "Top movers"]
)

with tab_brand:
    if SKU_COL and use_kategori:
        bdf = query_breakdown(unit, "Kategori", params)
        if bdf.empty:
            st.info("No data for current filters.")
        else:
            topn = 12
            show = bdf.head(topn).copy()
            show["pct"] = (show["share"] * 100).round(1)
            chart = alt.Chart(show).mark_bar().encode(
                y=alt.Y("label:N", sort='-x', axis=alt.Axis(labelAngle=0)),
                x=alt.X("qty:Q", title=f"Qty ({unit})")
            )
            txt = alt.Chart(show).mark_text(align="left", dx=6).encode(
                y=alt.Y("label:N", sort='-x'),
                x=alt.X("qty:Q"),
                text=alt.Text("pct:Q", format=".1f")
            )
            st.altair_chart((chart + txt).properties(height=380), use_container_width=True)
            st.dataframe(
                show[["label","qty","share"]]
                    .assign(share=lambda d: (d["share"]*100).round(1))
                    .rename(columns={"share":"share_%"}),
                use_container_width=True, height=320
            )
    else:
        st.info("Enable ‘Use kategori’ or map SKU column to view brand contribution.")

with tab_product:
    if SKU_COL:
        pdf = query_breakdown(unit, "SKU", params)
        if pdf.empty:
            st.info("No data.")
        else:
            topn = 20
            show = pdf.head(topn).copy()
            show["pct"] = (show["share"] * 100).round(1)
            chart = alt.Chart(show).mark_bar().encode(
                y=alt.Y("label:N", sort='-x', axis=alt.Axis(labelAngle=0)),
                x=alt.X("qty:Q", title=f"Qty ({unit})")
            )
            txt = alt.Chart(show).mark_text(align="left", dx=6).encode(
                y=alt.Y("label:N", sort='-x'),
                x=alt.X("qty:Q"),
                text=alt.Text("pct:Q", format=".1f")
            )
            st.altair_chart((chart + txt).properties(height=520), use_container_width=True)
    else:
        st.info("Map the SKU column to enable product contribution.")

with tab_channel:
    cdf = query_channel_share(unit, params)
    if cdf.empty:
        st.info("No data.")
    else:
        show = cdf.copy()
        show["pct"] = (show["share"] * 100).round(1)
        st.altair_chart(
            alt.Chart(show).mark_bar().encode(
                y=alt.Y("label:N", sort='-x', axis=alt.Axis(labelAngle=0)),
                x=alt.X("qty:Q", title=f"Qty ({unit})")
            ) + alt.Chart(show).mark_text(align="left", dx=6).encode(
                y="label:N", x="qty:Q", text=alt.Text("pct:Q", format=".1f")
            ),
            use_container_width=True
        )

with tab_region:
    if REGION_COL:
        rdf = query_breakdown(unit, "Region", params)
        if rdf.empty:
            st.info("No data.")
        else:
            show = rdf.head(20).copy()
            show["pct"] = (show["qty"] / rdf["qty"].sum() * 100).round(1)
            st.altair_chart(
                alt.Chart(show).mark_bar().encode(
                    y=alt.Y("label:N", sort='-x', axis=alt.Axis(labelAngle=0)),
                    x=alt.X("qty:Q", title=f"Qty ({unit})")
                ) + alt.Chart(show).mark_text(align="left", dx=6).encode(
                    y="label:N", x="qty:Q", text=alt.Text("pct:Q", format=".1f")
                ),
                use_container_width=True
            )
    else:
        st.info("Map the Region column to enable region contribution.")

with tab_movers:
    if SKU_COL:
        n_periods = 2
        mdf = query_breakdown_by_period(unit, "SKU", timegrain, n_periods, params)
        if mdf.empty or mdf["period"].nunique() < 2:
            st.info("Need at least two periods with data.")
        else:
            periods = sorted(mdf["period"].unique())
            last_key, prev_key = periods[-1], periods[-2]
            last_label = pretty_label_from_key(last_key, timegrain)
            prev_label = pretty_label_from_key(prev_key, timegrain)

            wide = (mdf.pivot_table(index="label", columns="period", values="qty", aggfunc="sum")
                        .fillna(0.0)
                        .assign(delta=lambda d: d[last_key] - d[prev_key]))
            up = wide.sort_values("delta", ascending=False).head(10).reset_index()
            dn = wide.sort_values("delta", ascending=True).head(10).reset_index()

            c1, c2 = st.columns(2)
            c1.subheader(f"Top gainers ({prev_label} → {last_label})")
            c1.altair_chart(
                alt.Chart(up).mark_bar().encode(
                    y=alt.Y("label:N", sort='-x', axis=alt.Axis(labelAngle=0)),
                    x=alt.X("delta:Q", title="Δ qty")
                ).properties(height=360),
                use_container_width=True
            )
            c2.subheader(f"Top decliners ({prev_label} → {last_label})")
            c2.altair_chart(
                alt.Chart(dn).mark_bar().encode(
                    y=alt.Y("label:N", sort='-x', axis=alt.Axis(labelAngle=0)),
                    x=alt.X("delta:Q", title="Δ qty")
                ).properties(height=360),
                use_container_width=True
            )
    else:
        st.info("Map the SKU column to see movers.")

# ---------- Table + Downloads ----------
st.subheader(f"{timegrain} table ({unit})")
tbl = agg[["label","value"]].rename(columns={"label": "period", "value": f"qty_{unit.lower()}"})
st.dataframe(tbl, use_container_width=True, height=360)

st.subheader("Downloads")
st.download_button(
    f"⬇️ Download {timegrain.lower()} CSV ({unit})",
    data=tbl.to_csv(index=False).encode("utf-8"),
    file_name=f"smsdv_{timegrain.lower()}_{unit.lower()}.csv",
    mime="text/csv",
)

buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    tbl.to_excel(writer, index=False, sheet_name=timegrain)
st.download_button(
    "⬇️ Download Excel",
    data=buf.getvalue(),
    file_name=f"smsdv_{timegrain.lower()}_{unit.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
