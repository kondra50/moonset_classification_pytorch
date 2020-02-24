import torch
import torch.nn as nn
from binary_classifier import Net,X,y,criterion,optimizer
#Initialize the model
model = Net()
# Number of epochs
epochs = 10000
#List to store losses
losses = []
for i in range(epochs):
    # Precit the output for Given input
    y_pred = model.forward(X)
    # Compute Cross entropy loss
    loss = criterion(y_pred,y)
    # Add loss to the list
    losses.append(loss.item())
    # Clear the previous gradients
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Adjust weights
    optimizer.step()

print(losses)