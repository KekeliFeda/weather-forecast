import pandas as pd
import torch
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from src.model import LSTMWeatherModel
from src.utils import WeatherInput, input_to_array, sequence_input_to_tensor
from datetime import datetime, timedelta
from .weather import get_historical_weather, get_weather_range  # Import from weather module

router = APIRouter()

# Load models at startup
lr_model = joblib.load("models/linear_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
lstm_model = LSTMWeatherModel()
lstm_model.load_state_dict(torch.load("models/lstm_model.pth", map_location=torch.device("cpu")))
lstm_model.eval()
  
@router.post("/linear")
def predict_linear(data: WeatherInput):
    try:
        x = input_to_array(data)
        pred = lr_model.predict(x)[0]
        return {"model": "Linear Regression", "prediction": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
     
@router.post("/random")
def predict_random(data: WeatherInput):
    try:
        x = input_to_array(data)
        pred = rf_model.predict(x)[0]
        return {"model": "Random Forest", "prediction": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/lstm")
def predict_lstm(data: list[WeatherInput]):
    try:
        x = sequence_input_to_tensor(data)
        
        # Load the scaler that was used during training
        scaler = joblib.load("models/lstm_scaler.pkl")
        
        with torch.no_grad():
            # Get raw predictions (these are in 0-1 scaled range)
            pred = lstm_model(x).squeeze().numpy()
            
            # Convert predictions back to actual temperature scale
            # Create dummy array with 8 features (matching training data structure)
            dummy_features = np.zeros((len(pred), 8))
            dummy_features[:, 0] = pred  # Put temperature predictions in first column (temperature feature)
            
            # Inverse transform to get actual temperature values
            unscaled_features = scaler.inverse_transform(dummy_features)
            actual_temps = unscaled_features[:, 0]  # Extract temperature column

        return {
            "model": "LSTM", 
            "prediction": [round(temp, 2) for temp in actual_temps.tolist()]
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/historical/{model_type}")
def predict_with_historical_context(model_type: str, date_time: str):
    """Make prediction using actual historical weather data as input"""
    # Get historical weather data
    weather_data = get_historical_weather(date_time)
    
    # Convert to the format expected by your models
    if model_type == 'lstm':
        # For LSTM, get 168 hours of historical data before the target datetime
        target_time = pd.to_datetime(date_time)
        start_time = target_time - timedelta(hours=167)
        
        range_data = get_weather_range(
            start_time.strftime('%Y-%m-%d %H:%M:%S'),
            date_time
        )
        
        if len(range_data['data']) < 168:
            raise HTTPException(status_code=400, detail="Insufficient historical data for LSTM prediction")
        
        # Use the range data for LSTM prediction
        lstm_input = []
        for item in range_data['data'][-168:]:  # Last 168 hours
            lstm_input.append({
                "temperature": item["temperature"],
                "precip": item["precip"],
                "snow": 0,  # Add default values for missing fields
                "humidity": item["humidity"],
                "rain": 0,
                "wind": item["wind"],
                "soil_temperature": item["temperature"] - 2,  # Approximate
                "soil_moist": 0.3  # Default value
            })
        
        return predict_lstm(lstm_input)
    
    else:
        # For linear/random forest, use single data point
        input_data = WeatherInput(**{
            k: v for k, v in weather_data.items() 
            if k in ['temperature', 'precip', 'snow', 'humidity', 'rain', 'wind', 'soil_temperature', 'soil_moist']
        })
        
        if model_type == 'linear':
            return predict_linear(input_data)
        elif model_type == 'random':
            return predict_random(input_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")