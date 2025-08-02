import numpy as np
from pydantic import BaseModel
import torch

class WeatherInput(BaseModel):
  temperature: float
  precip: float
  snow: float
  humidity: float 
  rain: float 
  wind: float
  soil_temperature: float
  soil_moist: float

def input_to_array(data: WeatherInput):
    return np.array([
       data.temperature,
       data.precip,
       data.snow,
       data.humidity,
       data.rain,
       data.wind,
       data.soil_temperature,
       data.soil_moist
    ]).reshape(1, -1)

def sequence_input_to_tensor(data: list[WeatherInput], input_steps: int = 168):
   if len(data) < input_steps:
      raise ValueError(f"LSTM requires at least {input_steps} hours of historical data")
   
   features = [
      [d.temperature, d.precip, d.snow, d.humidity, d.rain, d.wind, d.soil_temperature, d.soil_moist]
      for d in data[-input_steps:]
   ]

   tensor = torch.tensor([features], dtype=torch.float32)
   return tensor