import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

df = pd.read_csv('data/preprocessed_classical_weather.csv')

feature_cols = ['temperature', 'precip', 'snow', 'humidity', 'rain', 'wind', 'soil_temperature', 'soil_moist']
X = df[feature_cols].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#LINEAR REGRESSION MODEL

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

#RANDOM FOREST MODEL

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# METRICS FUNCTION

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    def directional_accuracy(actual, predicted):
        correct = 0
        total = 0
        for i in range(1, len(actual)):
            actual_trend = actual[i] - actual[i - 1]
            predicted_trend = predicted[i] - predicted[i - 1]
            if actual_trend * predicted_trend > 0:
                correct += 1
            total += 1
        return (correct / total) * 100 if total > 0 else 0
    
    def interval_accuracy(actual, predicted, tolerance=1):
        correct = sum(abs(a - p) <= tolerance for a, p in zip(actual, predicted))
        return (correct / len(actual)) * 100
    
    dir_acc = directional_accuracy(y_true, y_pred)
    int_acc = interval_accuracy(y_true, y_pred)

    print(f"\n=== {name} Evaluation ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
    print(f"Interval Accuracy: {int_acc:.2f}%")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/linear_regression_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
print("Models saved in the 'models/' folder.")
