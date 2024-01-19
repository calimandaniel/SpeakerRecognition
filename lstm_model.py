import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Custom LSTM-based neural network model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            output_size (int): Number of output classes.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the neural network.
        """
        out, _ = self.lstm(x)  # Ignore the hidden state, only take the output
        out = torch.relu(out[:, -1, :])  # Take the last output for each sequence
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out