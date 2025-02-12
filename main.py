import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM Neural Network Model
class RealTimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RealTimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Get the last time-step output
        return out, hidden

# Hyperparameters
input_size = 1      # One feature (e.g., time-series data)
hidden_size = 32    # LSTM hidden layer size
num_layers = 1      # Number of LSTM layers
output_size = 1     # Predicting one value
learning_rate = 0.01

# Initialize the model, loss function, and optimizer
model = RealTimeLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize LSTM hidden state
hidden_state = (
    torch.zeros(num_layers, 1, hidden_size).to(device),
    torch.zeros(num_layers, 1, hidden_size).to(device),
)

# Simulated real-time data stream
def get_new_data():
    """Simulate real-time data streaming (replace with actual sensor/API input)."""
    return random.uniform(0, 10), random.uniform(0, 10)  # (input, target)

# Real-time training and prediction loop
print("Starting real-time LSTM learning...")

while True:
    # Get new real-time data
    x_new, y_new = get_new_data()

    # Convert to tensor and move to GPU
    x_tensor = torch.tensor([[[x_new]]], dtype=torch.float32).to(device)  # Shape: (1, 1, input_size)
    y_tensor = torch.tensor([[y_new]], dtype=torch.float32).to(device)   # Shape: (1, output_size)

    # Forward pass (prediction)
    with torch.no_grad():  # No gradient needed for inference
        y_pred, hidden_state = model(x_tensor, hidden_state)

    # Compute loss
    loss = criterion(y_pred, y_tensor)

    # Backpropagation and optimization
    optimizer.zero_grad()
    y_pred, hidden_state = model(x_tensor, hidden_state)  # Forward pass again for training
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

    # Output results
    print(f"Input: {x_new:.2f}, Predicted: {y_pred.item():.2f}, Actual: {y_new:.2f}, Loss: {loss.item():.4f}")

    # Simulating real-time delay (adjust as per real-world scenario)
    time.sleep(1)  # Process new data every second