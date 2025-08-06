import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Global variable for historical data
historical_df = None

def load_historical_data():
    global historical_df
    if historical_df is None:
        try:
            historical_df = pd.read_csv("data/lefke_weather_data.csv")
            historical_df['time'] = pd.to_datetime(historical_df['time'])
            historical_df = historical_df.sort_values('time')
        except Exception as e:
            print(f"Error loading historical data: {e}")
    return historical_df

@router.get("/dates/range")
def get_date_range():
    """Get available date range for the dataset"""
    df = load_historical_data()
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    return {
        "start_date": df['time'].min().strftime('%Y-%m-%d'),
        "end_date": df['time'].max().strftime('%Y-%m-%d'),
        "location": "Lefke, Northern Cyprus",
        "total_records": len(df)
    }

@router.get("/weather/historical/{date_time}")
def get_historical_weather(date_time: str):
    """Get actual weather data for a specific date and time"""
    df = load_historical_data()
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    try:
        # Parse the datetime string
        target_time = pd.to_datetime(date_time)
        
        # Find the closest time match (within 1 hour)
        time_diff = abs(df['time'] - target_time)
        closest_idx = time_diff.idxmin()
        
        if time_diff.iloc[closest_idx] > pd.Timedelta(hours=1):
            raise HTTPException(status_code=404, detail="No weather data found for this date/time")
        
        row = df.iloc[closest_idx]
        
        return {
            "datetime": row['time'].isoformat(),
            "temperature": float(row['temperature_2m (°C)']),                    # ← Fixed
            "precip": float(row['precipitation (mm)']),
            "snow": float(row['snowfall (cm)']),
            "humidity": float(row['relative_humidity_2m (%)']),
            "rain": float(row['rain (mm)']),
            "wind": float(row['wind_speed_10m (km/h)']),
            "soil_temperature": float(row['soil_temperature_0_to_7cm (°C)']),    # ← Fixed
            "soil_moist": float(row['soil_moisture_0_to_7cm (m³/m³)'])           # ← Fixed
        }
                
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use YYYY-MM-DDTHH:MM:SS")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Column not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/weather/range/{start_date}/{end_date}")
def get_weather_range(start_date: str, end_date: str):
    """Get weather data for a date range (for charts)"""
    df = load_historical_data()
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    try:
        start_time = pd.to_datetime(start_date)
        end_time = pd.to_datetime(end_date)
        
        # Filter data for the date range
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        filtered_df = df.loc[mask]
        
        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data found for this date range")
        
        # Convert to list of dictionaries
        weather_data = []
        for _, row in filtered_df.iterrows():
            weather_data.append({
                "datetime": row['time'].isoformat(),
                "temperature": float(row['temperature_2m (°C)']),           # ← Fixed
                "precip": float(row['precipitation (mm)']),
                "humidity": float(row['relative_humidity_2m (%)']),
                "wind": float(row['wind_speed_10m (km/h)'])
            })
        
        return {
            "location": "Lefke, Northern Cyprus",
            "data": weather_data,
            "count": len(weather_data)
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")