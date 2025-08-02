import torch
import joblib
from fastapi import APIRouter, HTTPException
from src.model import LSTMWeatherModel
from src.utils import WeatherInput, input_to_array, sequence_input_to_tensor

router = APIRouter()

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
        with torch.no_grad():
            pred = lstm_model(x).squeeze().numpy()

        return {
            "model": "LSTM", 
            "prediction": [round(p, 2) for p in pred.tolist()]
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))