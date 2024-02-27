import torch
import torch.nn as nn
import torch.optim.adam as Adam

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define model, loss function, and optimizer
input_size = 5
output_size = 3
model = SimpleNN(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Dummy input data
input_data = torch.randn(10, input_size)

# Forward pass
output = model(input_data)

# Dummy target data
target = torch.randint(0, output_size, (10,))

# Calculate loss
loss = criterion(output, target)

# Backward pass (compute gradients)
optimizer.zero_grad()  # Clear previous gradients
loss.backward()       # Compute gradients
optimizer.step()       # Update weights and biases using the optimizer

# Print results
print("Forward pass output:")
print(output)
print("Categorical Cross-Entropy Loss:")
print(loss.item())
print("Gradients of the Loss with respect to Weights and Biases:")
for param in model.parameters():
    print(param.grad)
