import torch
import torch.nn as nn

class LSTMWeatherModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=24):
        super(LSTMWeatherModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to output the prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Get output from last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
