from fastapi import FastAPI
from api import predict 

app = FastAPI(
  title="Weather Prediction API",
  description="Serve predictions from LSTM, Linear Regression, and Random Forest models",
  version="1.0.0"
)

app.include_router(predict.router, prefix="/predict")

@app.get("/")
def read_root():
  return {"message": "Welcome to the Weather Prediction API!"}