# import streamlit as st


# st.write(
#   "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

import streamlit as st
import pandas as pd
import pickle
import numpy as np


# Load your pre-trained model from local storage (e.g., a pickle file)
@st.cache_resource
def load_model():
    with open('mc124_randomforest.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


model = load_model()

st.title("PM10 LEVEL OVERVIEW")


# ---- CHOOSE STATION TO PREDICT AND OVERVIEW ----
st.subheader(f"Prediction of pm10 for the current hour", divider="blue")


option = st.selectbox(
    "Which station would you like to get the data for?",
    ("Tempelhof-Schöneberg (mc124)", "IDK", "IDK 2"),
)
predicted_pm10 = 2.90
r2_score = 0.53

if predicted_pm10 > 20:
    status_text = f":red[UNHEALTHY]"
else:
    status_text = f":green[HEALTHY]"

# Display the string with conditional formatting
st.markdown(f"PM10 value for {option}: **{predicted_pm10:.2f} µg/m³** - **{status_text}**")

#st.write(f"**Current predicted PM10 value for {option}**: {predicted_pm10:.2f} µg/m³")
st.write(f"R2 score of predicted PM10 value: **{r2_score}**")
# Import your preprocessing script (if necessary)
# from data_processing import load_and_preprocess_data

# Sample Data Loading (You can replace this with your own function)

st.subheader(f"Overview of pm10 progression for {option}", divider="blue")

@st.cache_data
def load_data():
    df = pd.read_csv('df_h-1_complete_mc124.csv', parse_dates=['datetime'])
    return df


df = load_data()


# Sample Data Loading (You can replace this with your own function)
@st.cache_data
def load_data():
    df = pd.read_csv('df_h-1_complete_mc124.csv', parse_dates=['datetime'])
    return df


df = load_data()

# Sidebar for User Inputs
st.sidebar.header('View Options')
view = st.sidebar.selectbox('Select View', [
    'Yearly Overview (Comparison)',
    'Monthly Average (One Year)',
    'Weekly Overview (Daily Averages)',
    'Daily View (Hourly Averages)'
])

# Shared Date Filters
start_year = df['datetime'].dt.year.min()
end_year = df['datetime'].dt.year.max()

# Define each visualization case
if view == 'Yearly Overview (Comparison)':
    # Select multiple years for comparison
    selected_years = st.sidebar.multiselect(
        'Select Years to Compare',
        sorted(df['datetime'].dt.year.unique()),
        default=[2022, 2023, 2024]  # Set default years to show
    )

    # Filter data based on the selected years
    df_filtered = df[df['datetime'].dt.year.isin(selected_years)]

    # Group by year and calculate the yearly sum/mean
    df_grouped = df_filtered.groupby(df_filtered['datetime'].dt.year)['pm10'].mean().reset_index()

    # Rename columns for clearer chart display
    df_grouped.columns = ['Year', 'Average pm10']

    # Plotting without months or commas (just the years)
    st.line_chart(df_grouped.rename(columns={'Year': 'index'}).set_index('index'))

elif view == 'Monthly Average (One Year)':
    # Select a single year for the monthly average view
    selected_year = st.sidebar.selectbox(
        'Select Year',
        sorted(df['datetime'].dt.year.unique()),
        index=0
    )

    # Filter data based on the selected year
    df_filtered = df[df['datetime'].dt.year == selected_year]

    # Group by month and calculate the monthly average
    df_grouped = df_filtered.groupby(df_filtered['datetime'].dt.month)['pm10'].mean().reset_index()
    df_grouped.columns = ['Month', 'Average pm10']

    # Limit to exactly 12 months (no scrolling further)
    df_grouped = df_grouped[df_grouped['Month'] <= 12]

    # Plotting
    st.line_chart(df_grouped.rename(columns={'Month': 'index'}).set_index('index'))

elif view == 'Weekly Overview (Daily Averages)':
    # Select a year for the weekly view
    selected_year = st.sidebar.selectbox(
        'Select Year',
        sorted(df['datetime'].dt.year.unique()),
        index=0
    )

    # Filter data based on the selected year
    df_filtered = df[df['datetime'].dt.year == selected_year]

    # Group by week number and calculate the daily averages per week
    df_filtered['week'] = df_filtered['datetime'].dt.isocalendar().week
    df_grouped = df_filtered.groupby('week')['pm10'].mean().reset_index()
    df_grouped.columns = ['Week', 'Average pm10']

    # Limit to exactly 52 weeks (some years might have 53 weeks)
    df_grouped = df_grouped[df_grouped['Week'] <= 52]

    # Plotting
    st.line_chart(df_grouped.rename(columns={'Week': 'index'}).set_index('index'))

elif view == 'Daily View (Hourly Averages)':
    # Select a specific day for the hourly view
    selected_date = st.sidebar.date_input(
        'Select Date',
        value=pd.to_datetime('2020-01-01')  # Default date
    )

    # Filter data for the selected date
    df_filtered = df[df['datetime'].dt.date == selected_date]

    # Group by hour and calculate the hourly averages
    df_filtered['hour'] = df_filtered['datetime'].dt.hour
    df_grouped = df_filtered.groupby('hour')['pm10'].mean().reset_index()
    df_grouped.columns = ['Hour', 'Average pm10']

    # Limit to 24 hours
    df_grouped = df_grouped[df_grouped['Hour'] < 24]

    # Plotting
    st.line_chart(df_grouped.rename(columns={'Hour': 'index'}).set_index('index'))
