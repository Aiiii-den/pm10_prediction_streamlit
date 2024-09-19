import re
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import median_absolute_error, r2_score
import numpy as np

import api_calls_predictions
from api_calls_graphs import load_historical_data, fetch_latest_hour_data


# ---- ML MODEL STUFF ----

@st.cache_resource
def load_model():
    with open('mc124_randomforest.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


model = load_model()

# ---- VISUALISATION DATA LOADING ----

# Define the stations
stations = ["Tempelhof-Schöneberg (mc124)", "Wedding (mc010)"]
pattern = r'\(([^)]+)\)'


# Calls function from api_calls_graphs.py to load initial data
@st.cache_data()
def get_initial_data():
    all_data = {}
    for station in stations:
        match = re.search(pattern, station)
        all_data[station] = load_historical_data(match.group(1))
    return all_data


# Load initial data
if 'incremented_data' not in st.session_state:
    st.session_state['incremented_data'] = get_initial_data()


def update_data():
    all_data = st.session_state['incremented_data']
    for station in all_data.keys():
        latest_hour_data = fetch_latest_hour_data(station)
        if not latest_hour_data.empty:
            all_data[station] = pd.concat([all_data[station], latest_hour_data], ignore_index=True)
    st.session_state['incremented_data'] = all_data


# TODO FIX because I wont deploy on exact hour
# Update data periodically
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = time.time()

# Calculate elapsed time and refresh data if necessary
elapsed_time = time.time() - st.session_state['last_update']
if elapsed_time > 3600:  # 1 hour
    update_data()
    st.session_state['last_update'] = time.time()
    st.rerun()  # Refresh the Streamlit app

# ---- UI STUFF ----

st.title("PM10 LEVEL OVERVIEW")

# ---- CHOOSE STATION TO PREDICT AND OVERVIEW ----
st.subheader(f"Prediction of pm10 for the current hour", divider="blue")

chosen_station = st.selectbox(
    "Which station would you like to get the data for?",
    stations,
)

df = st.session_state['incremented_data'][chosen_station]
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce',
                                utc=True)  # TODO remove utc=True once datetime has been formatted

# input_data_prediction = pd.DataFrame({
#    'pm10_h-1': df["core"] == "pm10",
#    # Add more features as needed
# })

if st.button('Get prediction'):
    current_datetime = datetime.now()
    datetime_h_pred = current_datetime - timedelta(hours=1)
    datetime_h_pred = datetime_h_pred.replace(minute=0, second=0, microsecond=0)
    datetime_h_r2 = current_datetime - timedelta(hours=2)
    datetime_h_r2 = datetime_h_r2.replace(minute=0, second=0, microsecond=0)

    # regex the station code out of the chosen_station
    pattern = r'\(([^)]+)\)'
    match = re.search(pattern, chosen_station)

    input_pred = api_calls_predictions.fetch_weather_data(match.group(1), datetime_h_pred, datetime_h_pred)
    st.dataframe(input_pred)
    print(model.feature_names_in_)
    predicted_pm10 = model.predict(input_pred)[0]

    input_r2 = api_calls_predictions.fetch_weather_data(match.group(1), datetime_h_r2, datetime_h_r2)
    st.dataframe(input_r2)
    y_r2 = input_r2
    X_r2 = input_pred['pm10_h-1']
    predicted_h_prev = model.predict(y_r2)
    st.write(X_r2)
    st.write(predicted_h_prev[0])
    r2 = r2_score(X_r2, predicted_h_prev)


    if predicted_pm10 > 20:
        status_text = f":red[UNHEALTHY]"
    else:
        status_text = f":green[SAFE]"

    st.markdown(f"PM10 value for {chosen_station}: **{predicted_pm10:.2f} µg/m³** - **{status_text}**")
    st.write(f"Prediction vs actual pm10 value of the previous hour: **{predicted_h_prev[0]}** - **{X_r2} "
             f"| Median Absolute Error: {0}")


st.subheader(f"Overview of pm10 progression for {chosen_station}", divider="blue")

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
        default=[2022, 2023, 2024]  # Set default years to show
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
