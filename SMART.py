# smart_dashboard_enhanced.py

# ===== IMPORTS SECTION =====
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from io import BytesIO
import re
from datetime import datetime, date, timedelta
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
import base64
from streamlit_drawable_canvas import st_canvas
import urllib

# ===== PAGE CONFIG =====
st.set_page_config(page_title="SMART Dashboard", layout="wide", initial_sidebar_state="expanded")

# ===== CSS STYLING =====
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
    
    .info-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .section-header {
        background: #f5f5f5;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 15px 0;
        font-weight: 600;
        font-size: 18px;
    }
    
    .param-box {
        background: #f3e5f5;
        border: 1px solid #9c27b0;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
    }
    
    .styled-table th {
        background: #667eea;
        color: white;
        padding: 10px;
        text-align: left;
    }
    
    .styled-table td {
        padding: 8px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .red-cell {
        background-color: #ef5350 !important;
        color: white !important;
        font-weight: bold;
    }
    
    .date-available {
        background-color: #4caf50 !important;
        color: white !important;
    }
    
    .date-unavailable {
        background-color: #e0e0e0 !important;
        color: #999 !important;
    }
    
    @media print {
        .red-cell {
            background-color: #ef5350 !important;
            color: white !important;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
        .no-print {
            display: none !important;
        }
        /* Hide Address column when printing */
        .smi-table table thead tr th:last-child,
        .smi-table table tbody tr td:last-child {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE INIT =====
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_division' not in st.session_state:
    st.session_state.selected_division = ""
if 'selected_amo' not in st.session_state:
    st.session_state.selected_amo = ""
if 'selected_warehouse' not in st.session_state:
    st.session_state.selected_warehouse = ""
if 'selected_channel' not in st.session_state:
    st.session_state.selected_channel = ""
if 'selected_supervisor' not in st.session_state:
    st.session_state.selected_supervisor = ""
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = ""
if 'selected_salesman' not in st.session_state:
    st.session_state.selected_salesman = ""
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = None
if 'review_submitted' not in st.session_state:
    st.session_state.review_submitted = False
if 'available_dates' not in st.session_state:
    st.session_state.available_dates = []

# ===== DATABASE CONNECTION =====
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

# ===== HELPER FUNCTIONS =====
@st.cache_data(ttl=60)
def get_available_dates(region=None, warehouse=None):
    """Get dates that have data in SMI_Final"""
    query = "SELECT DISTINCT CAST(Sales_Date AS DATE) as Sales_Date FROM SMI_Final WHERE 1=1"
    params = {}
    
    if region:
        query += " AND UPPER(Region) = UPPER(:region)"
        params['region'] = region
    if warehouse:
        query += " AND (WH_Name = :warehouse OR WH_Name = :warehouse_wh OR WH_Name LIKE :warehouse_pattern)"
        params['warehouse'] = warehouse
        params['warehouse_wh'] = warehouse + '_WH'
        params['warehouse_pattern'] = warehouse + '%'
    
    query += " ORDER BY Sales_Date DESC"
    
    try:
        df = pd.read_sql(text(query), engine, params=params)
        return df['Sales_Date'].tolist()
    except Exception as e:
        st.warning(f"Could not fetch available dates: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_filtered_values(division=None, amo=None, depo=None, channel=None, supervisor=None):
    """Get cascading filtered values"""
    results = {
        'divisions': [],
        'amos': [],
        'depos': [],
        'channels': [],
        'supervisors': [],
        'regions': []
    }
    
    try:
        # Divisions
        results['divisions'] = pd.read_sql(
            text("SELECT DISTINCT Division FROM dakar WHERE Division IS NOT NULL ORDER BY Division"), 
            engine
        )['Division'].tolist()
        
        # AMOs filtered by Division
        amo_query = "SELECT DISTINCT AMO FROM dakar WHERE AMO IS NOT NULL"
        amo_params = {}
        if division:
            amo_query += " AND Division = :division"
            amo_params['division'] = division
        amo_query += " ORDER BY AMO"
        
        if amo_params:
            results['amos'] = pd.read_sql(text(amo_query), engine, params=amo_params)['AMO'].tolist()
        else:
            results['amos'] = pd.read_sql(text(amo_query), engine)['AMO'].tolist()
        
        # Depos filtered by Division and AMO
        depo_query = "SELECT DISTINCT Depo FROM dakar WHERE Depo IS NOT NULL"
        depo_params = {}
        if division:
            depo_query += " AND Division = :division"
            depo_params['division'] = division
        if amo:
            depo_query += " AND AMO = :amo"
            depo_params['amo'] = amo
        depo_query += " ORDER BY Depo"
        
        if depo_params:
            results['depos'] = pd.read_sql(text(depo_query), engine, params=depo_params)['Depo'].tolist()
        else:
            results['depos'] = pd.read_sql(text(depo_query), engine)['Depo'].tolist()
        
        # Channels filtered by Depo
        channel_query = "SELECT DISTINCT Channel FROM dakar WHERE Channel IS NOT NULL"
        channel_params = {}
        if depo:
            channel_query += " AND Depo = :depo"
            channel_params['depo'] = depo
        channel_query += " ORDER BY Channel"
        
        if channel_params:
            results['channels'] = pd.read_sql(text(channel_query), engine, params=channel_params)['Channel'].tolist()
        else:
            results['channels'] = pd.read_sql(text(channel_query), engine)['Channel'].tolist()
        
        # Supervisors filtered by Depo and Channel
        spv_query = "SELECT DISTINCT Employee_Name_1 as Supervisor FROM dakar WHERE Employee_Name_1 IS NOT NULL"
        spv_params = {}
        if depo:
            spv_query += " AND Depo = :depo"
            spv_params['depo'] = depo
        if channel:
            spv_query += " AND Channel = :channel"
            spv_params['channel'] = channel
        spv_query += " ORDER BY Employee_Name_1"
        
        if spv_params:
            results['supervisors'] = pd.read_sql(text(spv_query), engine, params=spv_params)['Supervisor'].tolist()
        else:
            results['supervisors'] = []
        
        # Regions filtered by Depo and Supervisor
        region_query = "SELECT DISTINCT Region FROM dakar WHERE Region IS NOT NULL"
        region_params = {}
        if depo:
            region_query += " AND Depo = :depo"
            region_params['depo'] = depo
        if supervisor:
            region_query += " AND Employee_Name_1 = :supervisor"
            region_params['supervisor'] = supervisor
        region_query += " ORDER BY Region"
        
        if region_params:
            results['regions'] = pd.read_sql(text(region_query), engine, params=region_params)['Region'].tolist()
        else:
            results['regions'] = pd.read_sql(text(region_query), engine)['Region'].tolist()
        
    except Exception as e:
        st.error(f"Error fetching filter values: {str(e)}")
    
    return results

@st.cache_data(ttl=60)
def get_smi_data(region, sales_date, warehouse=None):
    """Get SMI_Final data"""
    if isinstance(sales_date, str):
        date_str = sales_date
    else:
        date_str = sales_date.strftime('%Y-%m-%d')
    
    query = """
    SELECT 
        Region,
        WH_Name,
        Sales_Date,
        CAST(Sequence AS INT) as Sequence,
        Visit_Time,
        Customer_Code,
        Store_Name,
        Visit_Div_,
        ISNULL(Pack_Regular, 0) as Pack_Regular,
        ISNULL(Pack_NPL, 0) as Pack_NPL,
        Employee_Name_1,
        Channel,
        Division,
        Sales_Office_Name,
        Distance_Outlet_m,
        Latitude,
        Longitude,
        Outlet_Latitude,
        Outlet_Longitude,
        Outlet_Address,
        Radius_MNT_m,
        Google_Maps,
        Travel_Time,
        Transaction_Travel_Time,
        Fake_Indication
    FROM SMI_Final
    WHERE UPPER(Region) = UPPER(:region)
    AND CAST(Sales_Date AS DATE) = :sales_date
    """
    
    params = {'region': region, 'sales_date': date_str}
    
    if warehouse:
        query += " AND (WH_Name = :warehouse OR WH_Name = :warehouse_wh OR WH_Name LIKE :warehouse_pattern)"
        params['warehouse'] = warehouse
        params['warehouse_wh'] = warehouse + '_WH'
        params['warehouse_pattern'] = warehouse + '%'
    
    query += " ORDER BY Sequence"
    
    try:
        df = pd.read_sql(text(query), engine, params=params)
        return df
    except Exception as e:
        st.error(f"Error fetching SMI data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_cmec_data(region, sales_date, amo=None):
    """Get CMEC data"""
    if isinstance(sales_date, str):
        date_str = sales_date
    else:
        date_str = sales_date.strftime('%Y-%m-%d')
    
    query = """
    SELECT 
        ISNULL(SUM(CAST(Tetap_outlet AS FLOAT)), 0) as Target_Call,
        ISNULL(SUM(CAST(CM_Tetap_1_ AS FLOAT)), 0) as CM_Tetap,
        ISNULL(SUM(CAST(EC_Tetap_2_ AS FLOAT)), 0) as EC_Tetap,
        ISNULL(SUM(CAST(CM_Dummy_3_ AS FLOAT)), 0) as CM_Dummy,
        ISNULL(SUM(CAST(EC_Dummy_4_ AS FLOAT)), 0) as EC_Dummy,
        ISNULL(SUM(CAST(Sales_Qty_ AS FLOAT)), 0) as Sales_Qty
    FROM cmec
    WHERE Region_Code = :region
    AND CAST(Sales_Date AS DATE) = :sales_date
    """
    params = {'region': region, 'sales_date': date_str}
    
    if amo:
        query += " AND Org_Name = :amo"
        params['amo'] = amo
    
    try:
        df = pd.read_sql(text(query), engine, params=params)
        if not df.empty:
            return {
                'Target_Call': int(df.iloc[0]['Target_Call']),
                'CM_Total': int(df.iloc[0]['CM_Tetap'] + df.iloc[0]['CM_Dummy']),
                'EC_Total': int(df.iloc[0]['EC_Tetap'] + df.iloc[0]['EC_Dummy']),
                'Sales_Qty': float(df.iloc[0]['Sales_Qty'])
            }
    except Exception as e:
        st.warning(f"CMEC data not available: {str(e)}")
    
    return {'Target_Call': 0, 'CM_Total': 0, 'EC_Total': 0, 'Sales_Qty': 0}

def get_channel_parameters(channel):
    """Get channel-specific parameters"""
    params = {
        "Retail": {
            "working_time_min": 360,  # 6 hours
            "working_time_max": 480,  # 8 hours
            "first_outlet": "09:30:00",
            "tt_min": 600,  # 10 min in seconds
            "tt_max": 900,  # 15 min in seconds
            "radius_limit": 300,
            "distance_limit": 2000,
            "last_outlet_min": "16:00:00",
            "last_outlet_max": "17:00:00"
        },
        "Wholesaler": {
            "working_time_min": 360,
            "working_time_max": 480,
            "first_outlet": "09:00:00",
            "tt_min": 900,  # 15 min
            "tt_max": 1800,  # 30 min
            "radius_limit": 300,
            "distance_limit": 3000,
            "last_outlet_min": "16:00:00",
            "last_outlet_max": "17:00:00"
        },
        "WHOLESALER": {
            "working_time_min": 360,
            "working_time_max": 480,
            "first_outlet": "09:00:00",
            "tt_min": 900,
            "tt_max": 1800,
            "radius_limit": 300,
            "distance_limit": 3000,
            "last_outlet_min": "16:00:00",
            "last_outlet_max": "17:00:00"
        },
        "MMI": {
            "working_time_min": 360,
            "working_time_max": 480,
            "first_outlet": "09:00:00",
            "tt_min": 900,
            "tt_max": 1800,
            "radius_limit": 300,
            "distance_limit": 2500,
            "last_outlet_min": "16:00:00",
            "last_outlet_max": "17:00:00"
        },
        "Horeca": {
            "working_time_min": 360,
            "working_time_max": 480,
            "first_outlet": "09:30:00",
            "tt_min": 600,
            "tt_max": 1200,  # 20 min
            "radius_limit": 300,
            "distance_limit": 2000,
            "last_outlet_min": "16:00:00",
            "last_outlet_max": "17:00:00"
        }
    }
    return params.get(channel, params["Retail"])

def apply_red_marks(df, channel):
    """Apply red background to cells that violate standards"""
    if df.empty:
        return df
    
    params = get_channel_parameters(channel)
    
    def highlight_row(row):
        styles = [''] * len(row)
        
        # Distance violation
        if 'Distance_Outlet_m' in row.index:
            distance_val = pd.to_numeric(row['Distance_Outlet_m'], errors='coerce')
            if pd.notna(distance_val) and distance_val > params['distance_limit']:
                col_idx = row.index.get_loc('Distance_Outlet_m')
                styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
        
        # Radius violation
        if 'Radius_MNT_m' in row.index:
            radius_val = pd.to_numeric(row['Radius_MNT_m'], errors='coerce')
            if pd.notna(radius_val) and radius_val > params['radius_limit']:
                col_idx = row.index.get_loc('Radius_MNT_m')
                styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
        
        # Pack violations (null or 0)
        if 'Pack' in row.index:
            pack_val = pd.to_numeric(row['Pack'], errors='coerce')
            if pd.isna(pack_val) or pack_val == 0:
                col_idx = row.index.get_loc('Pack')
                styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
        
        if 'Pack_NPL' in row.index:
            pack_npl_val = pd.to_numeric(row['Pack_NPL'], errors='coerce')
            if pd.isna(pack_npl_val) or pack_npl_val == 0:
                col_idx = row.index.get_loc('Pack_NPL')
                styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
        
        # Visit Div violation (contains "Fixed")
        if 'Visit_Div_' in row.index:
            if pd.notna(row['Visit_Div_']) and 'Fixed' in str(row['Visit_Div_']):
                col_idx = row.index.get_loc('Visit_Div_')
                styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
        
        # Travel Time violation
        if 'Travel_Time' in row.index and pd.notna(row['Travel_Time']):
            try:
                if isinstance(row['Travel_Time'], str):
                    parts = row['Travel_Time'].split(':')
                    tt_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    tt_seconds = row['Travel_Time'].total_seconds()
                
                if tt_seconds < params['tt_min'] or tt_seconds > params['tt_max']:
                    col_idx = row.index.get_loc('Travel_Time')
                    styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
            except:
                pass
        
        # Transaction & Travel Time violation
        if 'Transaction_Travel_Time' in row.index and pd.notna(row['Transaction_Travel_Time']):
            try:
                if isinstance(row['Transaction_Travel_Time'], str):
                    parts = row['Transaction_Travel_Time'].split(':')
                    ttt_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    ttt_seconds = row['Transaction_Travel_Time'].total_seconds()
                
                if ttt_seconds < params['tt_min'] or ttt_seconds > params['tt_max']:
                    col_idx = row.index.get_loc('Transaction_Travel_Time')
                    styles[col_idx] = 'background-color: #ffcdd2; font-weight: bold;'
            except:
                pass
        
        return styles
    
    return df.style.apply(highlight_row, axis=1)

@st.cache_data(ttl=60)
def get_submitted_review(region, sales_date):
    """Get submitted review for region and date"""
    if isinstance(sales_date, str):
        date_str = sales_date
    else:
        date_str = sales_date.strftime('%Y-%m-%d')
    
    query = """
    SELECT TOP 1
        Region,
        SalesDate,
        Supervisor,
        SmartFeedback,
        OdmsFeedback,
        PhotoPath,
        SpvSignature,
        SalesmanSignature,
        CreatedAt
    FROM spv_reviews
    WHERE UPPER(LTRIM(RTRIM(Region))) = UPPER(LTRIM(RTRIM(:region)))
    AND CAST(SalesDate AS DATE) = CAST(:sales_date AS DATE)
    ORDER BY CreatedAt DESC
    """
    
    try:
        df = pd.read_sql(text(query), engine, params={'region': region, 'sales_date': date_str})
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception as e:
        # Table might not exist yet
        pass
    
    return None

# ===== HEADER =====
st.markdown('<div class="header-title">💡 S.M.A.R.T - Sales Monitoring And Route Tracking</div>', unsafe_allow_html=True)

# ===== SIDEBAR FILTERS =====
st.sidebar.markdown("### 🔍 Filters (Required)")

# Division
filter_values = get_filtered_values()
division = st.sidebar.selectbox(
    "Division*", 
    [""] + filter_values['divisions'],
    key="division_filter"
)

# AMO (cascading from Division)
amos = []
if division:
    filter_values = get_filtered_values(division=division)
    amos = filter_values['amos']

amo = st.sidebar.selectbox(
    "AMO*", 
    [""] + amos,
    key="amo_filter"
)

# Warehouse (cascading from Division and AMO)
depos = []
if division or amo:
    filter_values = get_filtered_values(division=division, amo=amo)
    depos = filter_values['depos']

warehouse = st.sidebar.selectbox(
    "Warehouse/Depo*", 
    [""] + depos,
    key="warehouse_filter"
)

# Channel (cascading from Warehouse)
channels = []
supervisors = []
regions = []
if warehouse:
    filter_values = get_filtered_values(division=division, amo=amo, depo=warehouse)
    channels = filter_values['channels']

channel = st.sidebar.selectbox(
    "Channel*", 
    [""] + channels,
    key="channel_filter"
)

# Supervisor (cascading from Warehouse and Channel)
if warehouse and channel:
    filter_values = get_filtered_values(division=division, amo=amo, depo=warehouse, channel=channel)
    supervisors = filter_values['supervisors']

supervisor = st.sidebar.selectbox(
    "Supervisor*", 
    [""] + supervisors,
    key="supervisor_filter"
)

# Region (cascading from Warehouse and Supervisor)
if warehouse and supervisor:
    filter_values = get_filtered_values(division=division, amo=amo, depo=warehouse, channel=channel, supervisor=supervisor)
    regions = filter_values['regions']

region = st.sidebar.selectbox(
    "Region*", 
    [""] + regions,
    key="region_filter"
)

# Get available dates
available_dates = []
if region and warehouse:
    available_dates = get_available_dates(region, warehouse)
    st.session_state.available_dates = available_dates
    
    if available_dates:
        st.sidebar.success(f"✅ {len(available_dates)} dates with data available")
        # Show last 5 available dates
        st.sidebar.caption("Recent dates with data:")
        for d in available_dates[:5]:
            st.sidebar.caption(f"• {d.strftime('%Y-%m-%d')}")
    else:
        st.sidebar.warning("⚠️ No data available for selected filters")
else:
    # Use session state if available
    if st.session_state.available_dates:
        available_dates = st.session_state.available_dates

# Date selection
default_date = available_dates[0] if available_dates else date(2025, 9, 19)
selected_date = st.sidebar.date_input(
    "Sales Date*",
    value=default_date,
    key="date_filter"
)

# Show if selected date has data
if available_dates and selected_date in available_dates:
    st.sidebar.success("✅ Data available for selected date")
elif available_dates:
    st.sidebar.error("❌ No data for this date. Please select from available dates above.")

# View Mode Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 View Mode")
view_mode = st.sidebar.radio(
    "Select View:",
    ["Dynamic (Fast)", "Static (Print-Ready)"],
    help="Dynamic: Fast loading, no colors. Static: All rows visible, colored violations for printing."
)

# Load button
if st.sidebar.button("✅ Load Data", use_container_width=True, type="primary"):
    if all([division, amo, warehouse, channel, supervisor, region, selected_date]):
        st.session_state.data_loaded = True
        st.rerun()
    else:
        st.sidebar.error("Please fill all required fields!")

# Check if should display data
if not st.session_state.data_loaded:
    st.info("📝 Please select all required filters and click 'Load Data' to view data.")
    st.stop()

# ===== MAIN LAYOUT =====
col1, col2, col3 = st.columns([1.5, 3, 1.5])

# NPL Products
npl_products = [
    "Juara Berry 12",
    "Tango Orangeyogo 12", 
    "W7N Click Ice Mango 16",
    "Juara Apel",
    "W7N Click Juicy 16"
]

# Left column - Standard Information
with col1:
    st.markdown('<div class="section-header">Standard Information</div>', unsafe_allow_html=True)
    
    info_html = f"""
    <div class="info-card">
        <table style="width:100%">
            <tr><td><b>Date:</b></td><td>{selected_date}</td></tr>
            <tr><td><b>Division:</b></td><td>{division}</td></tr>
            <tr><td><b>AMO:</b></td><td>{amo}</td></tr>
            <tr><td><b>Warehouse:</b></td><td>{warehouse}</td></tr>
            <tr><td><b>Channel:</b></td><td>{channel}</td></tr>
            <tr><td><b>Supervisor:</b></td><td>{supervisor}</td></tr>
            <tr><td><b>Region:</b></td><td>{region}</td></tr>
            <tr><td colspan="2"><hr></td></tr>
            <tr><td colspan="2"><b>NPL Products:</b></td></tr>
            <tr><td colspan="2" style="font-size:11px">{"<br>".join(npl_products)}</td></tr>
        </table>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)

# Get SMI data
df_smi = get_smi_data(region, selected_date, warehouse)
cmec_data = get_cmec_data(region, selected_date, amo)

# Middle column
with col2:
    # Sales Monitoring
    st.markdown('<div class="section-header">Sales Monitoring</div>', unsafe_allow_html=True)
    
    if not df_smi.empty:
        target_call = cmec_data['Target_Call'] if cmec_data['Target_Call'] > 0 else 40
        cm_total = cmec_data['CM_Total'] if cmec_data['CM_Total'] > 0 else len(df_smi[df_smi['Visit_Div_'] == 'sale']) if 'Visit_Div_' in df_smi.columns else len(df_smi)
        ec_total = len(df_smi[pd.to_numeric(df_smi['Pack_NPL'], errors='coerce') > 0])
        ec_npl = len(df_smi[(pd.to_numeric(df_smi['Pack_NPL'], errors='coerce') > 0) & 
                            (~df_smi['Customer_Code'].astype(str).str.startswith('9'))])
        volume_total = int(pd.to_numeric(df_smi['Pack_Regular'], errors='coerce').sum() + 
                          pd.to_numeric(df_smi['Pack_NPL'], errors='coerce').sum())
        volume_npl = int(pd.to_numeric(df_smi['Pack_NPL'], errors='coerce').sum())
        drop_size = round(volume_total / ec_total, 1) if ec_total > 0 else 0
        sku_active = df_smi['Pack_NPL'].nunique()
    else:
        target_call = cmec_data['Target_Call']
        cm_total = cmec_data['CM_Total']
        ec_total = cmec_data['EC_Total']
        ec_npl = 0
        volume_total = int(cmec_data['Sales_Qty'])
        volume_npl = 0
        drop_size = 0
        sku_active = 0
    
    sales_html = f"""
    <table class="styled-table">
        <thead>
            <tr>
                <th>Outlet Type</th>
                <th>Target Call</th>
                <th>CM</th>
                <th>EC Total</th>
                <th>EC NPL</th>
                <th>Drop Size</th>
                <th>Volume Total</th>
                <th>Volume NPL</th>
                <th>SKU Active</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Registered</td>
                <td>{target_call}</td>
                <td>{cm_total}</td>
                <td>{ec_total}</td>
                <td>{ec_npl}</td>
                <td>{drop_size}</td>
                <td>{volume_total}</td>
                <td>{volume_npl}</td>
                <td rowspan="2">{sku_active}</td>
            </tr>
            <tr>
                <td>Dummy</td>
                <td>-</td>
                <td>0</td>
                <td>0</td>
                <td>0</td>
                <td>0</td>
                <td>0</td>
                <td>0</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(sales_html, unsafe_allow_html=True)
    
    # Route Tracking
    st.markdown('<div class="section-header">Route Tracking</div>', unsafe_allow_html=True)
    
    if not df_smi.empty:
        # Calculate metrics
        try:
            visit_times = pd.to_datetime(df_smi['Visit_Time'], format='%H:%M:%S', errors='coerce')
            working_seconds = (visit_times.max() - visit_times.min()).total_seconds()
            working_hours = int(working_seconds // 3600)
            working_minutes = int((working_seconds % 3600) // 60)
            working_secs = int(working_seconds % 60)
            working_time_str = f"{working_hours}:{working_minutes:02d}:{working_secs:02d}"
        except:
            working_time_str = "N/A"
        
        first_outlet = df_smi['Visit_Time'].min() if 'Visit_Time' in df_smi.columns else "N/A"
        last_outlet = df_smi['Visit_Time'].max() if 'Visit_Time' in df_smi.columns else "N/A"
        fake_indication = len(df_smi[pd.to_numeric(df_smi['Fake_Indication'], errors='coerce') > 1])
        radius_gt_300 = len(df_smi[pd.to_numeric(df_smi['Radius_MNT_m'], errors='coerce') > 300])
        avg_distance = int(pd.to_numeric(df_smi['Distance_Outlet_m'], errors='coerce').mean())
        
        # Calculate average travel time
        try:
            tt_values = []
            for tt in df_smi['Travel_Time'].dropna():
                try:
                    if isinstance(tt, str):
                        # Handle string format "HH:MM:SS"
                        parts = tt.split(':')
                        if len(parts) == 3:
                            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            tt_values.append(seconds)
                    elif hasattr(tt, 'total_seconds'):
                        # Handle timedelta object
                        tt_values.append(int(tt.total_seconds()))
                    elif hasattr(tt, 'hour'):
                        # Handle time object
                        seconds = tt.hour * 3600 + tt.minute * 60 + tt.second
                        tt_values.append(seconds)
                except (ValueError, AttributeError):
                    continue
            
            if tt_values and len(tt_values) > 0:
                avg_tt_seconds = int(np.mean(tt_values))
                avg_tt = f"{avg_tt_seconds//3600:02d}:{(avg_tt_seconds%3600)//60:02d}:{avg_tt_seconds%60:02d}"
            else:
                avg_tt = "00:00:00"
        except Exception as e:
            avg_tt = "00:00:00"
        
        # Check violations
        params = get_channel_parameters(channel)
        working_violation = ""
        first_outlet_violation = ""
        last_outlet_violation = ""
        avg_tt_violation = ""
        
        if working_seconds < params['working_time_min'] * 60 or working_seconds > params['working_time_max'] * 60:
            working_violation = " class='red-cell'"
        
        if fake_indication > 2:
            fake_violation = " class='red-cell'"
        else:
            fake_violation = ""
        
        if radius_gt_300 > 0:
            radius_violation = " class='red-cell'"
        else:
            radius_violation = ""
    else:
        working_time_str = first_outlet = last_outlet = avg_tt = "N/A"
        fake_indication = radius_gt_300 = avg_distance = 0
        working_violation = first_outlet_violation = last_outlet_violation = ""
        fake_violation = radius_violation = avg_tt_violation = ""
    
    route_html = f"""
    <table class="styled-table">
        <thead>
            <tr>
                <th>Working Time</th>
                <th>First Outlet</th>
                <th>Same Coordinate</th>
                <th>Avg Travel Time</th>
                <th>Radius > 300</th>
                <th>Avg Distance</th>
                <th>Last Outlet</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td{working_violation}>{working_time_str}</td>
                <td{first_outlet_violation}>{first_outlet}</td>
                <td{fake_violation}>{fake_indication}</td>
                <td{avg_tt_violation}>{avg_tt}</td>
                <td{radius_violation}>{radius_gt_300}</td>
                <td>{avg_distance}m</td>
                <td{last_outlet_violation}>{last_outlet}</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(route_html, unsafe_allow_html=True)

# Right column - Parameters
with col3:
    st.markdown('<div class="section-header">Parameters</div>', unsafe_allow_html=True)
    params = get_channel_parameters(channel)
    
    param_html = f"""
    <div class="param-box">
        <b>{channel} Standards:</b><br><br>
        • <b>Working Time:</b> {params['working_time_min']//60}-{params['working_time_max']//60} hours<br>
        • <b>First Outlet:</b> ≤ {params['first_outlet']}<br>
        • <b>TT Standard:</b> {params['tt_min']//60}-{params['tt_max']//60} min<br>
        • <b>Radius MNT:</b> ≤ {params['radius_limit']}m<br>
        • <b>Distance:</b> ≤ {params['distance_limit']/1000:.1f}km<br>
        • <b>Last Outlet:</b> {params['last_outlet_min']}-{params['last_outlet_max']}
    </div>
    """
    st.markdown(param_html, unsafe_allow_html=True)

# ===== VISIT DETAILS TABLE =====
st.markdown('<div class="section-header">Visit Details</div>', unsafe_allow_html=True)

if not df_smi.empty:
    # Show view mode info
    if view_mode == "Static (Print-Ready)":
        st.info("📄 Static View: All rows visible with red violation highlights. Optimized for printing.")
        
        # Calculate violation counts
        params = get_channel_parameters(channel)
        
        # Prepare display dataframe
        display_cols = ['Sequence', 'Visit_Time', 'Customer_Code', 'Store_Name', 
                        'Visit_Div_', 'Pack_Regular', 'Pack_NPL', 'Travel_Time',
                        'Distance_Outlet_m', 'Radius_MNT_m', 'Fake_Indication']
        
        available_cols = [col for col in display_cols if col in df_smi.columns]
        display_df = df_smi[available_cols].copy()
        
        # Format Pack columns to remove decimals and commas
        if 'Pack_Regular' in display_df.columns:
            display_df['Pack_Regular'] = pd.to_numeric(display_df['Pack_Regular'], errors='coerce').fillna(0).astype(int)
        if 'Pack_NPL' in display_df.columns:
            display_df['Pack_NPL'] = pd.to_numeric(display_df['Pack_NPL'], errors='coerce').fillna(0).astype(int)
        
        # Apply red mark styling
        styled_df = apply_red_marks(display_df, channel)
        st.dataframe(styled_df, use_container_width=True, height=None)  # None = show all rows
        
    else:  # Dynamic (Fast)
        st.info("⚡ Dynamic View: Fast loading, paginated view without color highlighting.")
        
        display_cols = ['Sequence', 'Visit_Time', 'Customer_Code', 'Store_Name', 
                        'Visit_Div_', 'Pack_Regular', 'Pack_NPL', 'Travel_Time',
                        'Distance_Outlet_m', 'Radius_MNT_m', 'Fake_Indication']
        
        available_cols = [col for col in display_cols if col in df_smi.columns]
        display_df = df_smi[available_cols].copy()
        
        # Format Pack columns to remove decimals and commas
        if 'Pack_Regular' in display_df.columns:
            display_df['Pack_Regular'] = pd.to_numeric(display_df['Pack_Regular'], errors='coerce').fillna(0).astype(int)
        if 'Pack_NPL' in display_df.columns:
            display_df['Pack_NPL'] = pd.to_numeric(display_df['Pack_NPL'], errors='coerce').fillna(0).astype(int)
        
        st.dataframe(display_df, use_container_width=True, height=400)
else:
    st.warning("No visit data found for selected filters")

# ===== MAP SECTION =====
st.markdown('<div class="section-header">🗺️ Route Map</div>', unsafe_allow_html=True)

if not df_smi.empty and 'Latitude' in df_smi.columns:
    df_map = df_smi.dropna(subset=['Latitude', 'Longitude']).copy()
    
    if not df_map.empty:
        df_map['Latitude'] = pd.to_numeric(df_map['Latitude'], errors='coerce')
        df_map['Longitude'] = pd.to_numeric(df_map['Longitude'], errors='coerce')
        df_map = df_map.dropna(subset=['Latitude', 'Longitude'])
        
        if not df_map.empty:
            if 'Sequence' in df_map.columns:
                df_map = df_map.sort_values('Sequence')
            
            center_lat = df_map['Latitude'].mean()
            center_lon = df_map['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
            
            # Add route path
            salesman_coords = list(zip(df_map['Latitude'], df_map['Longitude']))
            if len(salesman_coords) > 1:
                AntPath(
                    locations=salesman_coords,
                    delay=800,
                    dash_array=[10, 20],
                    color='#d32f2f',
                    weight=3,
                    opacity=0.8,
                    pulse_color='#ff6b6b'
                ).add_to(m)
            
            # Add markers
            for idx, row in df_map.iterrows():
                seq = int(row['Sequence']) if pd.notna(row['Sequence']) else idx + 1
                
                # Check violations for this outlet
                has_violation = False
                violation_text = []
                
                # Convert to numeric and check distance violation
                distance_val = pd.to_numeric(row.get('Distance_Outlet_m'), errors='coerce')
                if pd.notna(distance_val) and distance_val > 2000:
                    has_violation = True
                    violation_text.append(f"Distance: {distance_val:.0f}m")
                
                # Convert to numeric and check radius violation
                radius_val = pd.to_numeric(row.get('Radius_MNT_m'), errors='coerce')
                if pd.notna(radius_val) and radius_val > 300:
                    has_violation = True
                    violation_text.append(f"Radius: {radius_val:.0f}m")
                
                marker_color = '#ef5350' if has_violation else '#d32f2f'
                fill_color = '#ff6b6b' if has_violation else '#ef5350'
                
                # Salesman visit marker (LARGER SIZE)
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=12,  # Increased from 8
                    popup=folium.Popup(
                        f"""<b>Visit #{seq}</b><br>
                        Store: {row.get('Store_Name', 'N/A')}<br>
                        Code: {row.get('Customer_Code', 'N/A')}<br>
                        Time: {row.get('Visit_Time', 'N/A')}<br>
                        Status: {row.get('Visit_Div_', 'N/A')}<br>
                        {'<br><span style="color:red;font-weight:bold">⚠️ VIOLATIONS:<br>' + '<br>'.join(violation_text) + '</span>' if has_violation else ''}""",
                        max_width=300
                    ),
                    tooltip=f"Visit #{seq}: {row.get('Store_Name', 'Store')}" + (" ⚠️" if has_violation else ""),
                    color=marker_color,
                    fill=True,
                    fillColor=fill_color,
                    fillOpacity=0.8
                ).add_to(m)
                
                # Sequence number
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="
                            font-size: 16px;
                            color: white;
                            background-color: {marker_color};
                            border-radius: 50%;
                            width: 30px;
                            height: 30px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: bold;
                            border: 3px solid white;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        ">{seq}</div>
                        """
                    )
                ).add_to(m)
                
                # Outlet marker (LARGER SIZE)
                if pd.notna(row.get('Outlet_Latitude')) and pd.notna(row.get('Outlet_Longitude')):
                    outlet_lat = pd.to_numeric(row['Outlet_Latitude'], errors='coerce')
                    outlet_lon = pd.to_numeric(row['Outlet_Longitude'], errors='coerce')
                    
                    if pd.notna(outlet_lat) and pd.notna(outlet_lon):
                        folium.CircleMarker(
                            location=[outlet_lat, outlet_lon],
                            radius=14,  # Increased from 10
                            popup=folium.Popup(
                                f"""<b>Outlet #{seq}</b><br>
                                Store: {row.get('Store_Name', 'N/A')}<br>
                                Code: {row.get('Customer_Code', 'N/A')}<br>
                                Address: {row.get('Outlet_Address', 'N/A')}<br>
                                Radius: {row.get('Radius_MNT_m', 'N/A')}m""",
                                max_width=300
                            ),
                            tooltip=f"Outlet #{seq}: {row.get('Store_Name', 'Store')}",
                            color='#1976d2',
                            fill=True,
                            fillColor='#42a5f5',
                            fillOpacity=0.7
                        ).add_to(m)
                        
                        # Outlet sequence number (larger)
                        folium.Marker(
                            location=[outlet_lat, outlet_lon],
                            icon=folium.DivIcon(
                                html=f"""
                                <div style="
                                    font-size: 14px;
                                    color: white;
                                    background-color: #1976d2;
                                    border-radius: 50%;
                                    width: 26px;
                                    height: 26px;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    font-weight: bold;
                                    border: 3px solid white;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                                ">{seq}</div>
                                """
                            )
                        ).add_to(m)
                        
                        # Connection line
                        folium.PolyLine(
                            locations=[[row['Latitude'], row['Longitude']], [outlet_lat, outlet_lon]],
                            color='#757575',
                            weight=1,
                            opacity=0.5,
                            dash_array='5, 10'
                        ).add_to(m)
            
            # Legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 220px; height: auto; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid grey; border-radius: 5px; padding: 10px">
                <p style="margin: 5px;"><b>Map Legend</b></p>
                <p style="margin: 5px;"><span style="color: #d32f2f;">●</span> Salesman Route</p>
                <p style="margin: 5px;"><span style="color: #ef5350;">●</span> Violation Point</p>
                <p style="margin: 5px;"><span style="color: #1976d2;">●</span> Outlet Location</p>
                <p style="margin: 5px;">--- Salesman to Outlet</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            st_folium(m, height=500, use_container_width=True)
else:
    st.info("No map data available")

# ===== REVIEW SECTION =====
st.markdown("---")
st.markdown('<div class="section-header">📝 Reviews & Feedback</div>', unsafe_allow_html=True)

# Check if review already exists for this region and date
existing_review = get_submitted_review(region, selected_date)

if existing_review:
    # Display existing review
    st.success("✅ Review has been submitted for this region and date")
    
    col_review1, col_review2 = st.columns([2, 1])
    
    with col_review1:
        st.markdown("### 📋 Submitted Feedback")
        
        review_html = f"""
        <div class="info-card">
            <p><strong>Region:</strong> {existing_review.get('Region', 'N/A')}</p>
            <p><strong>Date:</strong> {existing_review.get('SalesDate', 'N/A')}</p>
            <p><strong>Supervisor:</strong> {existing_review.get('Supervisor', 'N/A')}</p>
            <p><strong>Submitted:</strong> {existing_review.get('CreatedAt', 'N/A')}</p>
            <hr>
            <p><strong>SMART Feedback:</strong></p>
            <p style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
                {existing_review.get('SmartFeedback', 'No feedback provided')}
            </p>
            <p><strong>ODMS Feedback:</strong></p>
            <p style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
                {existing_review.get('OdmsFeedback', 'No feedback provided')}
            </p>
        </div>
        """
        st.markdown(review_html, unsafe_allow_html=True)
    
    with col_review2:
        st.markdown("### ✍️ Signatures")
        if existing_review.get('SpvSignature'):
            st.success("✅ SPV Signature: Recorded")
        else:
            st.info("ℹ️ SPV Signature: Not recorded")
        
        if existing_review.get('SalesmanSignature'):
            st.success("✅ Salesman Signature: Recorded")
        else:
            st.info("ℹ️ Salesman Signature: Not recorded")
        
        if existing_review.get('PhotoPath'):
            st.success("📷 Photo: Attached")
        else:
            st.info("📷 Photo: Not attached")
    
    # Option to submit new review
    st.markdown("---")
    if st.button("📝 Submit New Review (Override)", type="secondary"):
        st.session_state.force_new_review = True
        st.rerun()

elif st.session_state.get('force_new_review', False) or not existing_review:
    # Show review form
    with st.form("review_form"):
        review_cols = st.columns([2, 1])
        
        with review_cols[0]:
            st.subheader("📝 Feedback")
            smart_feedback = st.text_area("SMART Feedback:", height=80, 
                                         placeholder="Enter feedback about sales monitoring and route tracking...")
            odms_feedback = st.text_area("ODMS Feedback:", height=80,
                                        placeholder="Enter feedback about outlet data management...")
        
        with review_cols[1]:
            st.subheader("📷 Photo")
            photo_option = st.radio("Select:", ["Upload", "Camera"])
            if photo_option == "Upload":
                review_photo = st.file_uploader("Upload Photo", type=['png', 'jpg', 'jpeg'])
            else:
                review_photo = st.camera_input("Take Photo")
        
        st.subheader("✍️ Digital Signatures")
        sig_cols = st.columns(2)
        
        with sig_cols[0]:
            st.write("**SPV Signature:**")
            spv_canvas = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#000000",
                background_color="#ffffff",
                height=150,
                width=300,
                drawing_mode="freedraw",
                key="spv_canvas",
                display_toolbar=True
            )
        
        with sig_cols[1]:
            st.write("**Salesman Signature:**")
            salesman_canvas = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#000000",
                background_color="#ffffff",
                height=150,
                width=300,
                drawing_mode="freedraw",
                key="salesman_canvas",
                display_toolbar=True
            )
        
        submit_cols = st.columns([1, 1, 1])
        with submit_cols[1]:
            if st.form_submit_button("📤 Submit Review", use_container_width=True, type="primary"):
                if smart_feedback or odms_feedback:
                    # Store review data
                    review_data = {
                        'date': selected_date,
                        'region': region,
                        'supervisor': supervisor,
                        'smart_feedback': smart_feedback,
                        'odms_feedback': odms_feedback,
                        'photo': review_photo,
                        'spv_signature': spv_canvas.json_data if spv_canvas else None,
                        'salesman_signature': salesman_canvas.json_data if salesman_canvas else None
                    }
                    
                    # Here you would save to database
                    # For now, just store in session state
                    st.session_state.last_review_data = review_data
                    st.session_state.review_submitted = True
                    st.session_state.force_new_review = False
                    
                    st.success("✅ Review submitted successfully!")
                    st.info("💾 Note: To persist reviews, connect to the spv_reviews database table")
                    st.rerun()
                else:
                    st.warning("⚠️ Please provide at least one feedback before submitting")

# ===== FOOTER =====
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
with col_f2:

    st.caption(f"🔒 SMART Dashboard | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Region: {region} | SPV: {supervisor}")
