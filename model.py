import torch

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 2)

    # This must be implemented
    def forward(self, x):
        # Output of the first layer
        x = self.fc1(x)
        # Activation function is Relu.
        x = F.relu(x)
        # Output of the second layer
        x = self.fc2(x)
        x = F.relu(x)
        # This produces output
        x = self.fc2(x)
        return x

    # This function takes an input and predicts the class, (0 or 1)
    def predict(self, x):
        # Apply softmax to output
        pred = F.softmax(self.forward(x))
        #pred = F.softmax(input, dim=2)
        #softmax(input, dim=3)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

model = Net()
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)