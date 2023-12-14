import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the LSTMModel.

        Parameters:
        - input_size: The number of expected features in the input x
        - hidden_size: The number of features in the hidden state h
        - output_size: The number of output features
        """
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Fully connected layer to map the hidden state to the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        - x: Input sequence of shape (batch_size, seq_length, input_size)

        Returns:
        - out: Output sequence of shape (batch_size, output_size)
        """
        # LSTM forward pass
        out, _ = self.lstm(x)
        
        # Use only the output from the last time step for the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out


# Dummy data for illustration
input_size = 10
hidden_size = 20
output_size = 1
seq_length = 5
batch_size = 32

# Instantiate the model
model = LSTMModel(input_size, hidden_size, output_size)

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

if torch.cuda.is_available():
    print("Model is using GPU.")
else:
    print("Model is using CPU.")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training data
# Input sequence of shape (batch_size, seq_length, input_size)
# Target sequence of shape (batch_size, output_size)
train_input = torch.randn(batch_size, seq_length, input_size).to(device)
train_target = torch.randn(batch_size, output_size).to(device)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_input)

    # Compute the loss
    loss = criterion(outputs, train_target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Dummy test data
# You should replace this with your actual test data
test_input = torch.randn(batch_size, seq_length, input_size).to(device)

# Test the model
model.eval()
with torch.no_grad():
    test_output = model(test_input)

# Print the test output (predictions)
print("Test Output:")
print(test_output)
