from random import sample
import streamlit as st
import pandas as pd
import numpy as np
import folium
from datetime import time
from folium import plugins
from folium.plugins import Draw
from streamlit_folium import st_folium
from PIL import Image
import geocoder

# 设置 matplotlib 参数（若有可视化需要）
rc = {'figure.figsize':(8,4.5),
      'axes.facecolor':'#0e1117',
      'axes.edgecolor': '#0e1117',
      'axes.labelcolor': 'white',
      'figure.facecolor': '#0e1117',
      'patch.edgecolor': '#0e1117',
      'text.color': 'white',
      'xtick.color': 'white',
      'ytick.color': 'white',
      'grid.color': 'grey',
      'font.size': 8,
      'axes.labelsize': 12,
      'xtick.labelsize': 8,
      'ytick.labelsize': 12}

slot2num = {'Midnight': 0, 'Morning': 1, 'Afternoon': 2, 'Night': 3}

def make_cluster_map(df, center, zoom, map_style):
    """
    A function to create a cluster map using folium.
    """
    chicago_cluster_map = folium.Map(location=center, zoom_start=zoom, tiles=map_style, 
                                     attr="Map data © OpenStreetMap contributors")
    marker_cluster = plugins.MarkerCluster().add_to(chicago_cluster_map)
    for _, row in df.iterrows():
        popup_text = f"""
            District: {row['District']}<br>
            Date: {row['Date']}<br>
            Description: {row['Primary Type']}<br>
            Location Description: {row['Location Description']}<br>
            Arrest: {row['Arrest']}<br>
        """
        popup = folium.Popup(popup_text, min_width=300, max_width=500)
        folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=popup, fill=True).add_to(marker_cluster)
    
    folium.GeoJson(
        'geojson/chicago.json',
        name='geojson',
        style_function=lambda feature: {'color': 'red', 'fillColor': 'red'}
    ).add_to(chicago_cluster_map)
    return chicago_cluster_map

def make_heat_map(df, center, zoom, map_style):
    """
    A function to create a heat map using folium.
    """
    chicago_heat_map = folium.Map(location=center, zoom_start=zoom, tiles=map_style, 
                                  attr="Map data © OpenStreetMap contributors")
    
    points = [[lat, lon] for lat, lon in zip(df['Latitude'], df['Longitude'])]
    
    folium.GeoJson(
        'geojson/chicago.json',
        name='geojson',
        style_function=lambda feature: {'color': 'red', 'fillColor': 'red'}
    ).add_to(chicago_heat_map)
    
    plugins.HeatMap(points).add_to(chicago_heat_map)
    return chicago_heat_map

def geo_coder(location):
    return geocoder.arcgis(location).latlng

# 数据加载和预处理
df = pd.read_csv('data/Chicago_crimes.csv').sample(10000)
df.drop(columns=['Unnamed: 0'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.index = pd.DatetimeIndex(df['Date'])

df['Month_n'] = pd.DatetimeIndex(df['Date']).month_name()
df['Day'] = pd.DatetimeIndex(df['Date']).day
df['Day_n'] = pd.DatetimeIndex(df['Date']).day_name()
df['Time'] = df['Date'].dt.time

sample = df.sample(5000)
lat, lon = df['Latitude'].mean(), df['Longitude'].mean()
center, zoom = [lat, lon], 10.5

def app(df=sample):
    st.title('Shadows in the Sun - The guilty secret in a busy city')
    image = Image.open('skyline.jpg')
    st.image(image, caption='Hell is empty. The devil is on Chicago.')

    # Selectors and data filtering
    year, month, week_day = df['Year'].unique(), df['Month_n'].unique(), df['Day_n'].unique()
    crime_type, district, location = df['Primary Type'].unique(), df['District'].unique(), df['Location Description'].unique()

     # Selectors
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.header("Time Selector")

    with col1_2:
        st.header("Category Selector")
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        slot_filter_year = st.multiselect(
            'Choosing the years',
            year,
            [2013,2014,2015,2016])

    with col2_2:
        slot_filter_type = st.multiselect(
            'Choosing the crime types',
            crime_type,
            ['THEFT','NARCOTICS'])

    col3_1, col3_2 = st.columns(2)
    with col3_1:
        slot_filter_month = st.multiselect(
            'Choosing the months',
            month,
            ['April','May','June','July'])
    with col3_2:
        slot_filter_district = st.multiselect(
            'Choosing the districts',
            district,
            [2.0, 3.0, 4.0, 5.0, 6.0]) 
            
    col4_1, col4_2 = st.columns(2)
    with col4_1:
        slot_filter_week_day = st.multiselect(
            'Choosing the week of days',
            week_day,
            ['Saturday','Sunday'])
    with col4_2:
        slot_filter_location = st.multiselect(
            'Choosing the location',
            location,
            ['STREET','APARTMENT','RESIDENCE'])

    # Select the Data
    if slot_filter_year:
        df = df[df.Year.isin(slot_filter_year)]

    if slot_filter_month:
        df = df[df.Month_n.isin(slot_filter_month)]

    if slot_filter_week_day:
        df = df[df.Day_n.isin(slot_filter_week_day)]

    if slot_filter_type:
        df = df[df['Primary Type'].isin(slot_filter_type)]

    if slot_filter_district:
        df = df[df.District.isin(slot_filter_district)]

    if slot_filter_location:
        df = df[df['Location Description'].isin(slot_filter_location)]

    # Summary Digital Panel
    st.header('Crime Digital Panel')
    col1, col2, col3 = st.columns(3)
    col1.metric("Case Number", f"{len(df)}")
    col2.metric("Under Arrest", f"{len(df[df['Arrest'] == True])}")
    if len(df) != 0:
        col3.metric("Arrest Rate", f"{len(df[df['Arrest'] == True])/len(df):.1%}")
    else:
        col3.metric("Arrest Rate", "0")        
    
    # Map selectors
    map_style = 'OpenStreetMap' if st.selectbox('Map Style', ('Normal', 'Black and White')) == 'Normal' else 'Stamen Toner'
    map_type = st.selectbox('Map Type', ('Heat Map', 'Cluster Map'))

    location_check = st.text_input('Enter a location to search', '')
    new_center = geo_coder(location_check) if location_check else None

    # Generating maps based on selection
    map_center, map_zoom = (new_center, 15) if new_center else (center, zoom)
    
    if map_type == 'Heat Map':
        heat_map = make_heat_map(df, map_center, map_zoom, map_style)
        Draw(export=True).add_to(heat_map)
        st.data = st_folium(heat_map, width=1500)
    else:
        cluster_map = make_cluster_map(df, map_center, map_zoom, map_style)
        Draw(export=True).add_to(cluster_map)
        st.data = st_folium(cluster_map, width=1500)

# 运行应用程序
app()