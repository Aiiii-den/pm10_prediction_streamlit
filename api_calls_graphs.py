import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# Fetch weather data for a specific month
def fetch_weather_data(station, start_date, end_date):
    params = {
        "timespan": "custom",
        "core": "pm10",
        "period": "1h",
        "start": start_date.strftime("%d.%m.%Y %H:%M"),
        "end": end_date.strftime("%d.%m.%Y %H:%M")
    }
    api_endpoint = f"https://luftdaten.berlin.de/api/stations/{station}/data"
    response = requests.get(api_endpoint, params=params)
    data = response.json()
    if data:
        month_data = pd.json_normalize(data)
        month_data = month_data.dropna()
        return cut_off_timezone_data(month_data)
    return pd.DataFrame()


# Function to load historical data for all months
def load_historical_data(station):
    historical_data = pd.DataFrame()
    start_date = datetime(2018, 1, 1)
    date_rn = datetime(2024, 9, 19)

    current_date = start_date
    while current_date <= date_rn:
        end_date = current_date + relativedelta(months=1) - timedelta(seconds=1)  # Last second of the month

        month_data = fetch_weather_data(station, current_date, end_date)

        historical_data = pd.concat([historical_data, month_data], ignore_index=True)

        current_date += relativedelta(months=1)

    return historical_data


def cut_off_timezone_data(month_data):
    month_data.loc[:, 'datetime'] = month_data['datetime'].astype(str).str.slice(0, 19)
    month_data['datetime'] = pd.to_datetime(month_data['datetime'], format='mixed')
    month_data.loc[:, 'datetime'] = month_data['datetime'].dt.tz_localize(None)
    print(month_data)
    return month_data
