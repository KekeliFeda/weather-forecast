import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# This function loads the weather CSV, cleans it, and scales it
def load_and_preprocess(filepath, save_path=None, scaler_path=None):
    df = pd.read_csv(filepath)

    # Convert 'time' to datetime and set it as index
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

    # Check and fill missing values (if any)
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    # Scale the features between 0 and 1
    features = ['temperature', 'precip', 'snow', 'humidity', 'rain', 'wind', 'soil_temperature', 'soil_moist']
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[features])

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

    df_scaled = pd.DataFrame(scaled_values, columns=features)
    df_scaled['time'] = df['time'].values

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_scaled.to_csv(save_path, index=False)
        print(f"Scaled data saved to: {save_path}")

    return df_scaled, scaler

def create_sequences(df, input_steps=168, output_steps=24):
    features = ['temperature', 'precip', 'snow', 'humidity', 'rain', 'wind', 'soil_temperature', 'soil_moist']
    data = df[features].values
    target = df['temperature'].values

    X, y =[], []

    for i in range(input_steps, len(data) - output_steps):
        X.append(data[i - input_steps:i])
        y.append(target[i:i + output_steps])

    return np.array(X), np.array(y)

if __name__ == "__main__":
    df_scaled, scaler = load_and_preprocess(
        filepath="data/lefke_weather_data.csv",
        save_path="data/preprocessed_lstm.csv",
        scaler_path="models/lstm_scaler.pkl"
    )

X, y = create_sequences(df_scaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("LSTM sequence generation and split complete.")