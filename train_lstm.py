import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src.model import LSTMWeatherModel

df = pd.read_csv("data/preprocessed_lstm.csv")

def create_sequences(df, input_steps=168, output_steps=24):
    features = ['temperature', 'precip', 'snow', 'humidity', 'rain', 'wind', 'soil_temperature', 'soil_moist']
    data = df[features].values
    target = df['temperature'].values

    X, y = [], []
    for i in range(input_steps, len(data) - output_steps):
        X.append(data[i - input_steps:i])
        y.append(target[i:i + output_steps])

    return np.array(X), np.array(y)

X, y = create_sequences(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

batch_size = 32
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMWeatherModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.numpy()

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
        if (actual_trend * predicted_trend) > 0:
            correct += 1
        total += 1
    return (correct / total) * 100 if total > 0 else 0
    
def interval_accuracy(actual, predicted, tolerance=1):
    correct = sum(abs(a - p) <= tolerance for a, p in zip(actual.flatten(), predicted.flatten()))
    return (correct / len(actual.flatten())) * 100
    
y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()

print("\n LSTM Evaluation Metrics ")
print(f"MSE: {mse:.4f}")
print(f"RMSE {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy(y_true_flat, y_pred_flat):.2f}%")
print(f"Interval Accuracy(±1°C): {interval_accuracy(y_true_flat, y_pred_flat):.2f}%")

torch.save(model.state_dict(), "models/lstm_model.pth")
print("\nLSTM model saved to models/lstm_model.pth")