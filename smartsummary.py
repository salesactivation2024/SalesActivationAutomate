# smart_summary.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from datetime import datetime, timedelta, date
from io import BytesIO
import urllib

st.set_page_config(page_title="SMART Summary", layout="wide", initial_sidebar_state="expanded")

# ============================ CSS Styling ============================
st.markdown("""
<style>
    .main { background: #f5f7fa; }
    
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    .section-header {
        background: #f5f5f5;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 15px 0;
        font-weight: 600;
        font-size: 18px;
    }
    
    .region-section {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .region-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 15px;
    }
    
    /* Review status badges */
    .reviewed-badge {
        background: #4caf50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        display: inline-block;
    }
    
    .not-reviewed-badge {
        background: #ff9800;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Performance indicators */
    .perf-good { 
        background-color: #c8e6c9 !important; 
        color: #2e7d32;
        font-weight: 600; 
    }
    
    .perf-warning { 
        background-color: #fff3e0 !important; 
        color: #f57c00;
        font-weight: 600; 
    }
    
    .perf-poor { 
        background-color: #ffcdd2 !important; 
        color: #c62828;
        font-weight: 600; 
    }
    
    /* Violation highlighting */
    .violation-cell {
        background-color: #ffcdd2 !important;
        color: #c62828 !important;
        font-weight: bold;
    }
    
    /* Table styling */
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        font-size: 11px;
    }
    
    .summary-table th {
        background: #667eea;
        color: white;
        padding: 8px 5px;
        border: 1px solid #5a67d8;
        text-align: center;
        font-size: 10px;
        font-weight: 600;
    }
    
    .summary-table th.date-header {
        background: #764ba2;
    }
    
    .summary-table th.review-header {
        background: #4caf50;
    }
    
    .summary-table td {
        padding: 5px;
        border: 1px solid #e0e0e0;
        text-align: center;
        font-size: 10px;
    }
    
    .summary-table tr:hover {
        background-color: #f5f5f5;
    }
    
    .review-cell {
        text-align: left !important;
        padding: 8px !important;
        max-width: 250px;
        font-size: 10px;
        line-height: 1.3;
    }
    
    .review-text {
        background: #f8f9fa;
        padding: 5px;
        border-radius: 3px;
        margin: 2px 0;
        border-left: 2px solid #667eea;
    }
    
    /* Stats cards */
    .stats-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stats-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stats-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================ Database Connection ============================
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

# ============================ Header ============================
st.markdown("""
<div class="header-title">
    SMART Summary Dashboard
    <p style="font-size: 16px; font-weight: 400; margin-top: 5px;">Sales Representative Performance by Region & Warehouse</p>
</div>
""", unsafe_allow_html=True)

# ============================ Filters ============================
with st.container():
    st.markdown('<div class="section-header">Filter Options</div>', unsafe_allow_html=True)
    
    filter_cols = st.columns([2, 2, 2, 1.5])
    
    with filter_cols[0]:
        date_range = st.date_input(
            "Date Range",
            value=(date.today() - timedelta(days=7), date.today()),
            key="date_range"
        )
    
    with filter_cols[1]:
        try:
            amo_query = "SELECT DISTINCT AMO FROM dakar WHERE AMO IS NOT NULL ORDER BY AMO"
            amo_df = pd.read_sql(text(amo_query), engine)
            amo_options = ["All AMOs"] + amo_df['AMO'].tolist()
        except:
            amo_options = ["All AMOs"]
        
        selected_amo = st.selectbox("AMO Filter", amo_options)
    
    with filter_cols[2]:
        try:
            channel_query = "SELECT DISTINCT Channel FROM dakar WHERE Channel IS NOT NULL ORDER BY Channel"
            channel_df = pd.read_sql(text(channel_query), engine)
            channel_options = ["All Channels"] + channel_df['Channel'].tolist()
        except:
            channel_options = ["All Channels"]
        
        selected_channel = st.selectbox("Channel", channel_options)
    
    with filter_cols[3]:
        show_violations = st.checkbox("Highlight Violations", value=True)

# ============================ Query Functions ============================
@st.cache_data(ttl=60)
def get_sr_summary_data(start_date, end_date, amo_filter, channel_filter):
    """Query SR performance data from SMI_Final"""
    
    query = """
    SELECT 
        s.Region,
        s.WH_Name as Warehouse,
        s.Employee_Name_1 as Supervisor,
        s.Channel,
        CAST(s.Sales_Date AS DATE) as Date,
        COUNT(DISTINCT s.Customer_Code) as Target_Call,
        COUNT(CASE WHEN s.Visit_Div_ = 'sale' THEN 1 END) as CM,
        COUNT(CASE WHEN ISNULL(s.Pack_NPL, 0) > 0 THEN 1 END) as EC_Total,
        COUNT(CASE WHEN ISNULL(s.Pack_NPL, 0) > 0 AND NOT s.Customer_Code LIKE '9%' THEN 1 END) as EC_NPL,
        CASE 
            WHEN COUNT(CASE WHEN ISNULL(s.Pack_NPL, 0) > 0 THEN 1 END) > 0 
            THEN SUM(ISNULL(s.Pack_Regular, 0) + ISNULL(s.Pack_NPL, 0)) / NULLIF(COUNT(CASE WHEN ISNULL(s.Pack_NPL, 0) > 0 THEN 1 END), 0)
            ELSE 0 
        END as Drop_Size,
        SUM(ISNULL(s.Pack_Regular, 0) + ISNULL(s.Pack_NPL, 0)) as Volume_Total,
        SUM(ISNULL(s.Pack_NPL, 0)) as Volume_NPL,
        COUNT(DISTINCT CASE WHEN ISNULL(s.Pack_NPL, 0) > 0 THEN s.Pack_NPL END) as SKU_Active,
        MIN(s.Visit_Time) as First_Outlet,
        MAX(s.Visit_Time) as Last_Outlet,
        AVG(TRY_CAST(s.Distance_Outlet_m AS FLOAT)) as Avg_Distance,
        COUNT(CASE WHEN TRY_CAST(s.Distance_Outlet_m AS FLOAT) > 2000 THEN 1 END) as Distance_Violations,
        COUNT(CASE WHEN TRY_CAST(s.Radius_MNT_m AS FLOAT) > 300 THEN 1 END) as Radius_Violations,
        COUNT(CASE WHEN s.Visit_Div_ LIKE '%Fixed%' THEN 1 END) as Fixed_Violations,
        COUNT(CASE WHEN ISNULL(s.Pack_Regular, 0) = 0 OR s.Pack_Regular IS NULL THEN 1 END) as Pack_Null_Count
    FROM SMI_Final s
    WHERE CAST(s.Sales_Date AS DATE) BETWEEN :start_date AND :end_date
    """
    
    params = {'start_date': start_date, 'end_date': end_date}
    
    if channel_filter != "All Channels":
        query += " AND s.Channel = :channel"
        params['channel'] = channel_filter
    
    query += """
    GROUP BY 
        s.Region, s.WH_Name, s.Employee_Name_1, s.Channel, CAST(s.Sales_Date AS DATE)
    ORDER BY s.Region, s.WH_Name, CAST(s.Sales_Date AS DATE)
    """
    
    try:
        df = pd.read_sql(text(query), engine, params=params)
        
        # Get review data - check if table exists first
        try:
            # Test if table exists
            test_query = "SELECT TOP 1 * FROM spv_reviews"
            pd.read_sql(text(test_query), engine)
            
            # Table exists, query it
            review_query = """
            SELECT 
                CAST(SalesDate AS DATE) as Date,
                UPPER(LTRIM(RTRIM(Region))) as Region,
                Supervisor,
                SmartFeedback,
                OdmsFeedback,
                PhotoPath,
                SpvSignature,
                SalesmanSignature,
                CreatedAt
            FROM spv_reviews
            WHERE CAST(SalesDate AS DATE) BETWEEN :start_date AND :end_date
            """
            
            df_reviews = pd.read_sql(text(review_query), engine, params=params)
            
            if not df_reviews.empty:
                # Standardize Region format for merging
                df['Region_Clean'] = df['Region'].str.upper().str.strip()
                df_reviews['Region_Clean'] = df_reviews['Region'].str.upper().str.strip()
                
                # Merge review data
                df = df.merge(
                    df_reviews[['Date', 'Region_Clean', 'Supervisor', 'SmartFeedback', 'OdmsFeedback', 'CreatedAt']], 
                    left_on=['Date', 'Region_Clean', 'Supervisor'],
                    right_on=['Date', 'Region_Clean', 'Supervisor'],
                    how='left'
                )
                
                df['Reviewed'] = df['SmartFeedback'].notna() | df['OdmsFeedback'].notna()
            else:
                df['Reviewed'] = False
                df['SmartFeedback'] = None
                df['OdmsFeedback'] = None
                df['CreatedAt'] = None
        except:
            # Table doesn't exist - add placeholder columns
            df['Reviewed'] = False
            df['SmartFeedback'] = None
            df['OdmsFeedback'] = None
            df['CreatedAt'] = None
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# ============================ Get Data ============================
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

df_summary = get_sr_summary_data(start_date, end_date, selected_amo, selected_channel)

# ============================ Summary Statistics ============================
if not df_summary.empty:
    st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)

    col_stats = st.columns(6)

    with col_stats[0]:
        total_regions = df_summary['Region'].nunique()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{total_regions}</div>
            <div class="stats-label">Total Regions</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats[1]:
        total_visits = int(df_summary['EC_Total'].sum())
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{total_visits}</div>
            <div class="stats-label">Total EC</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats[2]:
        avg_drop = df_summary['Drop_Size'].mean()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{avg_drop:.1f}</div>
            <div class="stats-label">Avg Drop Size</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats[3]:
        review_rate = (df_summary['Reviewed'].sum() / len(df_summary) * 100) if len(df_summary) > 0 else 0
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{review_rate:.0f}%</div>
            <div class="stats-label">Review Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats[4]:
        total_volume = int(df_summary['Volume_Total'].sum())
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{total_volume}</div>
            <div class="stats-label">Total Volume</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats[5]:
        total_violations = int(df_summary[['Distance_Violations', 'Radius_Violations', 'Fixed_Violations']].sum().sum())
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value" style="color: #ef5350;">{total_violations}</div>
            <div class="stats-label">Violations</div>
        </div>
        """, unsafe_allow_html=True)

    # ============================ Main Summary Table by Region & Warehouse ============================
    st.markdown('<div class="section-header">Performance Summary by Region & Warehouse</div>', unsafe_allow_html=True)

    # Group by Region
    for region in sorted(df_summary['Region'].unique()):
        region_data = df_summary[df_summary['Region'] == region]
        
        with st.container():
            st.markdown(f'<div class="region-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="region-header">Region: {region}</div>', unsafe_allow_html=True)
            
            # Group by Warehouse within Region
            for warehouse in sorted(region_data['Warehouse'].unique()):
                warehouse_data = region_data[region_data['Warehouse'] == warehouse]
                
                st.markdown(f"**Warehouse: {warehouse}**")
                
                # Get unique dates and supervisors
                unique_dates = sorted(warehouse_data['Date'].unique())
                unique_supervisors = sorted(warehouse_data['Supervisor'].unique())
                
                # Create HTML table
                html_table = '<table class="summary-table">'
                
                # Header rows
                html_table += '<thead><tr>'
                html_table += '<th rowspan="2">Supervisor</th>'
                html_table += '<th rowspan="2">Channel</th>'
                
                for date_val in unique_dates:
                    date_str = date_val.strftime('%m/%d')
                    colspan = 9 if show_violations else 8
                    html_table += f'<th colspan="{colspan}" class="date-header">{date_str}</th>'
                
                html_table += '<th rowspan="2" class="review-header" style="min-width:200px">Review Status & Result</th>'
                html_table += '</tr>'
                
                # Sub-header
                html_table += '<tr>'
                for _ in unique_dates:
                    html_table += '<th>TC</th><th>CM</th><th>EC</th><th>NPL</th><th>DS</th><th>VOL</th><th>SKU</th><th>Time</th>'
                    if show_violations:
                        html_table += '<th>Violations</th>'
                html_table += '</tr></thead><tbody>'
                
                # Data rows
                for spv in unique_supervisors:
                    spv_data = warehouse_data[warehouse_data['Supervisor'] == spv]
                    
                    if not spv_data.empty:
                        first_row = spv_data.iloc[0]
                        html_table += '<tr>'
                        html_table += f'<td style="text-align:left;font-weight:600">{spv}</td>'
                        html_table += f'<td>{first_row["Channel"]}</td>'
                        
                        # Collect review info for this row
                        review_info = []
                        
                        for date_val in unique_dates:
                            date_data = spv_data[spv_data['Date'] == date_val]
                            
                            if not date_data.empty:
                                row = date_data.iloc[0]
                                
                                # Target Call
                                tc_val = int(row['Target_Call'])
                                tc_class = 'perf-good' if tc_val >= 40 else 'perf-warning' if tc_val >= 30 else 'perf-poor'
                                html_table += f'<td class="{tc_class}">{tc_val}</td>'
                                
                                # CM
                                cm_val = int(row['CM'])
                                cm_class = 'perf-good' if cm_val >= 35 else 'perf-warning' if cm_val >= 25 else 'perf-poor'
                                html_table += f'<td class="{cm_class}">{cm_val}</td>'
                                
                                # EC Total
                                ec_val = int(row['EC_Total'])
                                ec_class = 'perf-good' if ec_val >= 30 else 'perf-warning' if ec_val >= 20 else 'perf-poor'
                                html_table += f'<td class="{ec_class}">{ec_val}</td>'
                                
                                # NPL
                                npl_val = int(row['EC_NPL'])
                                html_table += f'<td>{npl_val}</td>'
                                
                                # Drop Size
                                ds_val = round(row['Drop_Size'], 1)
                                html_table += f'<td>{ds_val}</td>'
                                
                                # Volume (no comma)
                                vol_val = int(row['Volume_Total'])
                                html_table += f'<td>{vol_val}</td>'
                                
                                # SKU
                                sku_val = int(row['SKU_Active'])
                                html_table += f'<td>{sku_val}</td>'
                                
                                # Working Time
                                first_time = str(row['First_Outlet'])[:5] if pd.notna(row['First_Outlet']) else "-"
                                last_time = str(row['Last_Outlet'])[:5] if pd.notna(row['Last_Outlet']) else "-"
                                time_str = f"{first_time}-{last_time}"
                                html_table += f'<td style="font-size:9px">{time_str}</td>'
                                
                                # Violations
                                if show_violations:
                                    total_violations = int(row.get('Distance_Violations', 0) + 
                                                         row.get('Radius_Violations', 0) + 
                                                         row.get('Fixed_Violations', 0))
                                    viol_class = 'violation-cell' if total_violations > 0 else ''
                                    html_table += f'<td class="{viol_class}">{total_violations if total_violations > 0 else "-"}</td>'
                                
                                # Collect review data
                                if row.get('Reviewed', False):
                                    date_str = date_val.strftime('%m/%d')
                                    review_text = f"<b>{date_str}:</b> "
                                    
                                    if pd.notna(row.get('SmartFeedback')):
                                        review_text += f"SMART: {row['SmartFeedback'][:50]}..."
                                    
                                    if pd.notna(row.get('OdmsFeedback')):
                                        if pd.notna(row.get('SmartFeedback')):
                                            review_text += " | "
                                        review_text += f"ODMS: {row['OdmsFeedback'][:50]}..."
                                    
                                    review_info.append(f'<div class="review-text">{review_text}</div>')
                            else:
                                # Empty cells for missing dates
                                html_table += '<td>-</td>' * (9 if show_violations else 8)
                        
                        # Add review column at the end
                        if review_info:
                            html_table += f'<td class="review-cell"><span class="reviewed-badge">REVIEWED</span>{"".join(review_info)}</td>'
                        else:
                            html_table += '<td class="review-cell"><span class="not-reviewed-badge">NOT REVIEWED</span></td>'
                        
                        html_table += '</tr>'
                
                html_table += '</tbody></table>'
                
                # Display table
                st.markdown(html_table, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ============================ Export Section ============================
    st.markdown('<div class="section-header">Export Options</div>', unsafe_allow_html=True)

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        if st.button("Export to Excel", use_container_width=True, type="primary"):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"SMART_Summary_{start_date}_to_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col_exp2:
        if st.button("Export to CSV", use_container_width=True, type="primary"):
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"SMART_Summary_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )

    with col_exp3:
        if st.button("Print View", use_container_width=True, type="primary"):
            st.info("Press Ctrl+P to print")

    # ============================ Database Setup Info ============================
    with st.expander("⚙️ Database Setup - Create Reviews Table"):
        st.markdown("""
        To enable review tracking, create the `spv_reviews` table in your database:
        
        ```sql
        CREATE TABLE spv_reviews (
            ReviewID INT IDENTITY(1,1) PRIMARY KEY,
            SalesDate DATE NOT NULL,
            Region VARCHAR(50) NOT NULL,
            Supervisor VARCHAR(100) NOT NULL,
            SmartFeedback NVARCHAR(MAX),
            OdmsFeedback NVARCHAR(MAX),
            PhotoPath VARCHAR(500),
            SpvSignature NVARCHAR(MAX),
            SalesmanSignature NVARCHAR(MAX),
            CreatedAt DATETIME DEFAULT GETDATE(),
            CONSTRAINT UQ_Review UNIQUE (SalesDate, Region, Supervisor)
        );
        
        -- Create index for faster queries
        CREATE INDEX IX_Reviews_Date_Region ON spv_reviews(SalesDate, Region);
        ```
        
        After creating the table, reviews submitted in the SMART Dashboard will be stored and displayed here.
        """)
        
        st.info("💡 Until the table is created, the summary will show 'NOT REVIEWED' for all entries.")

    # ============================ Footer ============================
    st.markdown("---")
    st.caption(f"""
        SMART Summary Dashboard | 
        Data Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} | 
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
else:

    st.warning("No data available for the selected filters. Please adjust your filters and try again.")
