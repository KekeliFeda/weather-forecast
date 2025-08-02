import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_classical(filepath, save_path=None):
  df = pd.read_csv(filepath)

  df['time'] = pd.to_datetime(df['time'])
  df = df.sort_values('time')

  df = df[[
    'time', 
    'temperature_2m (°C)', 
    'precipitation (mm)', 
    'snowfall (cm)', 
    'relative_humidity_2m (%)',
    'rain (mm)', 
    'wind_speed_10m (km/h)',
    'soil_temperature_0_to_7cm (°C)',
    'soil_moisture_0_to_7cm (m³/m³)'
  ]]

  df = df.rename(columns={
    'temperature_2m (°C)': 'temperature',
    'precipitation (mm)': 'precip',
    'snowfall (cm)': 'snow', 
    'relative_humidity_2m (%)': 'humidity',
    'rain (mm)': 'rain',
    'wind_speed_10m (km/h)': 'wind',
    'soil_temperature_0_to_7cm (°C)': 'soil_temperature',
    'soil_moisture_0_to_7cm (m³/m³)': 'soil_moist'
  })


  df.fillna(method='ffill', inplace=True)
  df.fillna(method='bfill', inplace=True)

  df['target'] = df['temperature'].shift(-1)

  df.dropna(inplace=True)

  feature_cols = ['temperature', 'precip', 'snow', 'humidity', 'rain', 'wind', 'soil_temperature', 'soil_moist']
  X = df[feature_cols]
  y = df['target']

  if save_path:
    df_to_save = df[['time'] + feature_cols + ['target']]
    df_to_save.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")

  return X, y

if __name__ == "__main__":
  X, y = load_and_preprocess_classical(
    filepath = "data/lefke_weather_data.csv",
    save_path="data/preprocessed_classical_weather.csv"
  )