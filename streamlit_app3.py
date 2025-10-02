import streamlit as st
import streamlit as st
import pandas as pd
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from io import BytesIO
import re
from datetime import timedelta
import streamlit as st

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




db = st.secrets.get("login", {})
username1   = db.get("username1",   "QWERT")
username2 = db.get("username2", "Streamlit123")
password1 = db.get("password1", "password1")
password2 = db.get("password2", "password2")


# Hardcoded credentials (username and password)
USER_CREDENTIALS = {
    f"{username1}": f"{password1}",
    f"{username2}": f"{password2}"
}

def login():
    import pandas as pd

    st.title("Login Page")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True  
            st.success(f"Welcome {username}!")
            st.experimental_set_query_params(page="afterlogin")  
        else:
            st.error("Invalid username or password")



def afterlogin():
    hal = [
        st.Page("SMART.py", title="S.M.A.R.T.", icon="💡"),
        st.Page("smartsummary.py", title="S.M.A.R.T. Summary & Result", icon="✅"),
        st.Page("SRpeformance.py", title="SR Performance", icon="🏅"),
        st.Page("Outletmanagement.py", title = "Outlet Management", icon = "🏪"),
        st.Page("bd.py", title = "Brand Distribution", icon = "🧮"),
        st.Page("weekly_comp.py", title="Weekly Comparation", icon="📆"),
        st.Page("Sales_Trend.py", title = "Sales Trend", icon = "📊"),
        
    ]

    nav = st.navigation(hal)

    nav.run()



def main():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login()  
    else:
        afterlogin()  

if __name__ == "__main__":

    main()
