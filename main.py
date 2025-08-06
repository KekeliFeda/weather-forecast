from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from api import predict, weather

app = FastAPI(
    title="Weather Prediction API",
    description="Serve predictions from LSTM, Linear Regression, and Random Forest models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/predict", tags=["predictions"])
app.include_router(weather.router, prefix="", tags=["weather-data"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Prediction API!"}