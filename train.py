import torch
import torch.nn as nn
import os
from model import Net
import sklearn.datasets


#plt.scatter(X[:,0],X[:,1],s=40,c=y)
#plt.show()
#plt.savefig('myfig.png' )

# numpy to tensor
X, y = sklearn.datasets.make_moons(500, noise=0.1)
X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)

def _get_train_loader():
    print("Get data loader.")
    X, y = sklearn.datasets.make_moons(500, noise=0.1)
    train_y = torch.from_numpy(y).type(torch.LongTensor)
    # features are the rest
    train_x = torch.from_numpy(X).type(torch.LongTensor)
    # create dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    return torch.utils.data.DataLoader(train_ds, batch_size=128)
def train(model, epochs, optimizer, criterion):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    total_loss = 0
    losses=[]
    for i in range(epochs):
        # Precit the output for Given inputloss
        y_pred = model.forward(X)
        # Compute Cross entropy loss
        loss = criterion(y_pred, y)
        # Add loss to the list
        losses.append(loss.item())
        # Clear the previous gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Adjust weights
        optimizer.step()
        total_loss += loss.item()
        print("Epoch: {}, Loss: {}".format(i,
                                           loss / len(X)))

    # save trained model, after all epochs
    save_model(model, 'model')

model = Net()
# Number of epochs

#List to store losses

if __name__ == '__main__':
    model = Net()

    # Given: save the parameters used to construct the model
    #save_model_params(model, args.model_dir)

    ## TODO: Define an optimizer and loss function for training
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #criterion = nn.BCELoss()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #train_loader = _get_train_loader()

    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, 100, optimizer, criterion)
