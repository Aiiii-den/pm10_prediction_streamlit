import requests
import pandas as pd
from datetime import datetime


def fetch_weather_data(station, start_date_time, end_date_time):
    start_date_time = start_date_time.strftime("%d.%m.%Y %H:%M")
    end_date_time = end_date_time.strftime("%d.%m.%Y %H:%M")
    params = {
        "timespan": "custom",
        "period": "1h",
        "start": start_date_time,
        "end": end_date_time
    }
    api_endpoint = f"https://luftdaten.berlin.de/api/stations/{station}/data"
    response = requests.get(api_endpoint, params=params)
    data = response.json()
    if data:
        return get_input_data(data)
    return pd.DataFrame()


def get_input_data(api_data):
    # Initialize an empty dictionary to hold the features
    features = {}

    # Parse the datetime from the first entry (assuming all entries have the same datetime)
    timestamp = pd.to_datetime(api_data[0]['datetime'])

    # Extract time features
    features['hour'] = timestamp.hour
    features['day'] = timestamp.day
    features['month'] = timestamp.month
    features['year'] = timestamp.year
    features['day_of_week'] = timestamp.weekday()  # Monday = 0, Sunday = 6
    features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0

    # Map to store pollutant values
    pollutant_map = {
        'no2_1h': 'no2_h-1',
        'no_1h': 'no_h-1',
        'nox_1h': 'nox_h-1',
        'pm10_1h': 'pm10_h-1',
        'pm2_1h': 'pm2.5_h-1'
    }

    # Loop through the API response to extract pollutant values
    for entry in api_data:
        component = entry['component']
        if component in pollutant_map:
            # Use the mapped feature name and assign the pollutant value
            features[pollutant_map[component]] = entry['value']

    # Convert the feature dictionary into a DataFrame (1 row, multiple columns)
    features_df = pd.DataFrame([features])
    expected_order_features = ['hour', 'day', 'month', 'year', 'day_of_week', 'is_weekend',
                               'no2_h-1', 'no_h-1', 'nox_h-1', 'pm10_h-1', 'pm2.5_h-1']
    features_df = features_df.reindex(columns=expected_order_features)
    return features_df
