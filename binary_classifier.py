import torch
import sklearn.datasets
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

X,y = sklearn.datasets.make_moons(500,noise=0.1)
plt.scatter(X[:,0],X[:,1],s=40,c=y)
#plt.show()
#plt.savefig('myfig.png' )

# numpy to tensor
X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)


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

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)