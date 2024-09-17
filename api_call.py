import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# Fetch weather data for a specific month
def fetch_weather_data(station, start_date, end_date):
    print(station)
    params = {
        "timespan": "custom",
        "core": "pm10",
        "period": "1h",
        "start": start_date.strftime("%d.%m.%Y %H:%M"),
        "end": end_date.strftime("%d.%m.%Y %H:%M")
    }
    api_endpoint = f"https://luftdaten.berlin.de/api/stations/{station}/data"
    print(api_endpoint)
    response = requests.get(api_endpoint, params=params)
    data = response.json()
    if data:
        month_data = pd.json_normalize(data) #TODO cut off hour +2 and stuff from datetime string
        month_data = month_data.dropna()
        print(month_data)
        return month_data
    return pd.DataFrame()


# Function to load historical data for all months
def load_historical_data(station):
    historical_data = pd.DataFrame()
    start_date = datetime(2022, 1, 1)
    date_rn = datetime(2024, 9, 17)
    print(start_date, date_rn)

    current_date = start_date
    while current_date <= date_rn:
        end_date = current_date + relativedelta(months=1) - timedelta(seconds=1)  # Last second of the month

        month_data = fetch_weather_data(station, current_date, end_date)

        historical_data = pd.concat([historical_data, month_data], ignore_index=True)

        current_date += relativedelta(months=1)

    return historical_data


# Function to fetch and append the latest hourly data
def fetch_latest_hour_data(station):
    now = datetime.now()
    start_date = now - timedelta(hours=1)
    end_date = now
    return fetch_weather_data(station, start_date, end_date)