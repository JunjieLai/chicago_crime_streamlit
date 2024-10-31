import streamlit as st
from multiapp import MultiApp
from apps import crimes_time_app, crimes_map_app

st.set_page_config(layout="wide")
apps = MultiApp()

# Add all your application here
apps.add_app("Time", crimes_time_app.app)
apps.add_app("Map", crimes_map_app.app)

# The main app
apps.run()