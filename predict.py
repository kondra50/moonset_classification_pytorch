import torch
from model import model
import sklearn.datasets
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

X, y = sklearn.datasets.make_moons(2000, noise=0.1)
def predict_fn(data,model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.from_numpy(X.astype('float32'))
    data = data.to(device)
    model.eval()
    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()
    return result

if __name__ == '__main__':
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    x = torch.from_numpy(X).type(torch.FloatTensor)
    ans = model.predict(x)
    print(ans.numpy())
    print(accuracy_score(model.predict(x), y))
    tp = np.logical_and(y, ans).sum()
    fp = np.logical_and(1 - y, ans).sum()
    tn = np.logical_and(1 - y, 1 - ans).sum()
    fn = np.logical_and(y, 1 - ans).sum()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(pd.crosstab(y, ans, rownames=['actuals'],
                      colnames=['predictions']))
    print("\n{:<11} {:.3f}".format('Recall:', recall))
    print("{:<11} {:.3f}".format('Precision:', precision))
    print("{:<11} {:.3f}".format('Accuracy:', accuracy))
    print()
