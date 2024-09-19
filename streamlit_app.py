import re
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import median_absolute_error
import numpy as np
import pytz

import api_calls_predictions
from api_calls_graphs import load_historical_data, fetch_weather_data


# ---- ML MODEL STUFF ----

@st.cache_resource
def load_model():
    with open('mc124_randomforest.pkl', 'rb') as f:
        model_load = pickle.load(f)
    return model_load


model = load_model()

# ---- VISUALISATION DATA LOADING ----

# Define the stations
stations = [
    "Buch (mc077) category: suburb",
    "Friedrichshain-Kreuzberg (mc174) category: traffic",
    "Grunewald (mc032) category: suburb",
    "Mitte (mc171) category: background",
    "Mitte (mc190) category: traffic",
    "Neuköln (mc042) category: background",
    "Neukölln (mc144) category: traffic",
    "Steglitz-Zehlendorf (mc117) category: traffic",
    "Tempelhof-Schöneberg (mc124) category: traffic",
    "Wedding (mc010) category: background",
]


# Calls function from api_calls_graphs.py to load initial data
@st.cache_data()
def get_initial_data():
    all_data = {}
    pattern_local = r'\(([^)]+)\)'
    for station_element in stations:
        match_initial = re.search(pattern_local, station_element)
        all_data[station_element] = load_historical_data(match_initial.group(1))
        print(station_element + " finished")
    return all_data


# Load initial data
if 'incremented_data' not in st.session_state:
    st.session_state['incremented_data'] = get_initial_data()


def update_data(station_name, start_date, end_date):
    all_data = st.session_state['incremented_data']
    match_initial = re.search(pattern, station_name)
    updated_data = fetch_weather_data(match_initial.group(1), start_date, end_date)
    if not updated_data.empty:
        all_data[station] = pd.concat([all_data[station], updated_data], ignore_index=True)
    st.session_state['incremented_data'] = all_data


# ---- TIMEZONE FIX ----
german_tz = pytz.timezone('Europe/Berlin')

# ---- UI STUFF ----

st.title("PM10 LEVEL OVERVIEW")

# ---- CHOOSE STATION TO PREDICT AND OVERVIEW ----
st.subheader(f"Prediction of pm10", divider="blue")

chosen_station = st.selectbox(
    "Which station would you like to get the data for?",
    stations,
)

df = st.session_state['incremented_data'][chosen_station]
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# regex the station code out of the chosen_station
pattern = r'\(([^)]+)\)'
match = re.search(pattern, chosen_station)
chosen_station_regex = match.group(1)
station_info_condensed = chosen_station[:chosen_station.index(')') + 1].strip()

if st.button('Get prediction'):
    current_datetime = datetime.now(german_tz)
    datetime_h_pred = current_datetime
    datetime_h_pred = datetime_h_pred.replace(minute=0, second=0, microsecond=0)
    datetime_h_mae = current_datetime - timedelta(hours=1)
    datetime_h_mae = datetime_h_mae.replace(minute=0, second=0, microsecond=0)

    input_pred = api_calls_predictions.fetch_weather_data(chosen_station_regex, datetime_h_pred, datetime_h_pred)
    if input_pred.empty:
        print(current_datetime)
        print("entered condition")
        datetime_h_pred = current_datetime - timedelta(hours=1)
        datetime_h_pred = datetime_h_pred.replace(minute=0, second=0, microsecond=0)
        datetime_h_mae = current_datetime - timedelta(hours=2)
        datetime_h_mae = datetime_h_mae.replace(minute=0, second=0, microsecond=0)
        input_pred = api_calls_predictions.fetch_weather_data(chosen_station_regex, datetime_h_pred, datetime_h_pred)

    predicted_pm10 = model.predict(input_pred)[0]

    input_mae = api_calls_predictions.fetch_weather_data(chosen_station_regex, datetime_h_mae, datetime_h_mae)
    y_mae = input_mae
    X_mae = input_pred['pm10_h-1'].item()
    predicted_h_prev = model.predict(y_mae)
    mae = median_absolute_error(np.array([X_mae]), predicted_h_prev)

    status_colors = {
        'good': '#006400',  # Dark Green
        'fair': '#008000',  # Green
        'moderate': '#FFFF00',  # Yellow
        'poor': '#FFA500',  # Orange
        'unhealthy': '#FF0000'  # Red
    }

    if predicted_pm10 < 21:
        colour = "#006400"
        status_text = "VERY GOOD"
    elif predicted_pm10 < 41:
        colour = "#90EE90"
        status_text = "GOOD"
    elif predicted_pm10 < 101:
        colour = "#FFFF00"
        status_text = "MODERATE"
    elif predicted_pm10 < 181:
        colour = "#FFA500"
        status_text = "BAD"
    else:
        colour = "#FF0000"
        status_text = "VERY BAD"

    proper_time_prediction = (datetime_h_pred + timedelta(hours=1)).hour

    st.markdown(
        f"Predicted PM10 value for {station_info_condensed} at {proper_time_prediction} o'clock: **{predicted_pm10:.2f} µg/m³** --"
        f" <span style='color:{colour};'><strong>{status_text}</strong></span>",
        unsafe_allow_html=True
    )
    st.write(f"Prediction vs actual pm10 value of the previous hour: **{predicted_h_prev[0]:.2f}** vs **{X_mae}**\n\n"
             f"Median Absolute Error of previous hour prediction: **{mae:.2f}**")

st.subheader(f"Overview of pm10 progression for {station_info_condensed}", divider="blue")

last_update_time = datetime.now(german_tz)
if st.button("Update Data"):
    datetime_from = last_update_time
    datetime_from = datetime_from.replace(minute=0, second=0, microsecond=0)
    datetime_till = datetime.now(german_tz)
    datetime_till = datetime_till.replace(minute=0, second=0, microsecond=0)
    for station in stations:
        update_data(station, datetime_from, datetime_till)
        last_update_time = datetime.now(german_tz)

formatted_update_time = last_update_time.strftime("%d.%m.%Y %H:%M")
st.write(f"Last updated: {formatted_update_time}")

# Sidebar for User Inputs
st.sidebar.header('View Options')
view = st.sidebar.selectbox('Select View', [
    'Yearly Comparison (Yearly Averages)',
    'Monthly Comparison (Monthly Averages)',
    'Weekly Comparison (Daily Averages)',
    'Daily Comparison (Hourly Averages)'
])

start_year = df['datetime'].dt.year.min()
end_year = df['datetime'].dt.year.max()

# Define each visualization case
if view == 'Yearly Comparison (Yearly Averages)':
    # Select multiple years for comparison
    selected_years = st.sidebar.multiselect(
        'Select Years to Compare',
        sorted(df['datetime'].dt.year.unique()),
        default=[2020, 2021, 2022, 2023, 2024]  # Set default years to show
    )
    # Filter data based on the selected years
    df_filtered = df[df['datetime'].dt.year.isin(selected_years)]

    # Group by year and calculate the yearly mean
    df_grouped = df_filtered.groupby(df_filtered['datetime'].dt.year)['value'].mean().reset_index()

    # Rename columns for clearer chart display
    df_grouped.columns = ['Year', 'Average pm10']

    # Convert 'Year' to string to ensure categorical x-axis
    df_grouped['Year'] = df_grouped['Year'].astype(str)

    # Plotting
    st.line_chart(df_grouped.set_index('Year'))


elif view == 'Monthly Comparison (Monthly Averages)':
    # Select a single year for the monthly average view
    selected_year = st.sidebar.selectbox(
        'Select Year',
        sorted(df['datetime'].dt.year.unique()),
        index=0
    )

    # Filter data based on the selected year
    df_filtered = df[df['datetime'].dt.year == selected_year]

    # Group by month and calculate the monthly average
    df_grouped = df_filtered.groupby(df_filtered['datetime'].dt.month)['value'].mean().reset_index()
    df_grouped.columns = ['Month', 'Average pm10']

    # Limit to exactly 12 months (no scrolling further)
    df_grouped = df_grouped[df_grouped['Month'] <= 12]

    # Plotting
    st.line_chart(df_grouped.rename(columns={'Month': 'index'}).set_index('index'))

if view == 'Weekly Comparison (Daily Averages)':
    # Select a year for the weekly view
    selected_year = st.sidebar.selectbox(
        'Select Year',
        sorted(df['datetime'].dt.year.unique()),
        index=0
    )

    # Select a week within the selected year
    selected_week = st.sidebar.selectbox(
        'Select Week',
        sorted(df[df['datetime'].dt.year == selected_year]['datetime'].dt.isocalendar().week.unique())
    )

    # Filter data based on the selected year and week
    df_filtered = df[
        (df['datetime'].dt.year == selected_year) &
        (df['datetime'].dt.isocalendar().week == selected_week)
        ]

    # Extract the day of the week
    df_filtered['day_of_week'] = df_filtered['datetime'].dt.day_name()

    # Group by day of the week and calculate daily averages
    df_grouped = df_filtered.groupby('day_of_week')['value'].mean().reset_index()

    # Sort days of the week to ensure the correct order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_grouped['day_of_week'] = pd.Categorical(df_grouped['day_of_week'], categories=days_order, ordered=True)
    df_grouped = df_grouped.sort_values('day_of_week')

    # Plotting
    st.line_chart(df_grouped.set_index('day_of_week'))


elif view == 'Daily Comparison (Hourly Averages)':
    # Select a specific day for the hourly view
    selected_date = st.sidebar.date_input(
        'Select Date',
        value=pd.Timestamp.now().date()
    )

    # Filter data for the selected date
    df_filtered = df[df['datetime'].dt.date == selected_date]

    # Group by hour and calculate the hourly averages
    df_filtered['hour'] = df_filtered['datetime'].dt.hour
    df_grouped = df_filtered.groupby('hour')['value'].mean().reset_index()
    df_grouped.columns = ['Hour', 'Average pm10']

    # Limit to 24 hours
    df_grouped = df_grouped[df_grouped['Hour'] < 24]

    # Plotting
    st.line_chart(df_grouped.rename(columns={'Hour': 'index'}).set_index('index'))
