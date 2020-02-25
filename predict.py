import torch
from model import model
import sklearn.datasets
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

X, y = sklearn.datasets.make_moons(500, noise=0.1)
def predict_fn(data,model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    data = torch.from_numpy(X.astype('float32'))
    data = data.to(device)

    # Put model into evaluation modeout = model(data)
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()  #.detach.numpy()
    #print(result)
    return result


# def evaluate(predictor, test_features, test_labels, verbose=True):
#     """
#     Evaluate a model on a test set given the prediction endpoint.
#     Return binary classification metrics.
#     :param predictor: A prediction endpoint
#     :param test_features: Test features
#     :param test_labels: Class labels for test data
#     :param verbose: If True, prints a table of all performance metrics
#     :return: A dictionary of performance metrics.
#     """
#
#     # rounding and squeezing array
#     test_preds = np.squeeze(np.round(predictor.predict(test_features)))
#
#     # calculate true positives, false positives, true negatives, false negatives
#     tp = np.logical_and(test_labels, test_preds).sum()
#     fp = np.logical_and(1 - test_labels, test_preds).sum()
#     tn = np.logical_and(1 - test_labels, 1 - test_preds).sum()
#     fn = np.logical_and(test_labels, 1 - test_preds).sum()
#
#     # calculate binary classification metrics
#     recall = tp / (tp + fn)
#     precision = tp / (tp + fp)
#     accuracy = (tp + tn) / (tp + fp + tn + fn)
#
#     # print metrics
#     if verbose:
#         print(pd.crosstab(test_labels, test_preds, rownames=['actuals'],
#                           colnames=['predictions']))
#         print("\n{:<11} {:.3f}".format('Recall:', recall))
#         print("{:<11} {:.3f}".format('Precision:', precision))
#         print("{:<11} {:.3f}".format('Accuracy:', accuracy))
#         print()
#
#     return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
#             'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}


if __name__ == '__main__':
    print("test")
    #train_x = torch.from_numpy(X).type(torch.LongTensor)
    #pr=predict_fn(train_x,model)
    #test_preds = np.squeeze(np.round(pr, y))
    #print(pr[0])
    #X = Variable(torch.from_numpy())
    #model.predict(X)
    x = torch.from_numpy(X).type(torch.FloatTensor)
    ans = model.predict(x)
    #print(ans.numpy())
    #print(accuracy_score(model.predict(X), y))
    #return ans.numpy()
    tp = np.logical_and(y, ans).sum()
    fp = np.logical_and(1 - y, ans).sum()
    tn = np.logical_and(1 - y, 1 - ans).sum()
    fn = np.logical_and(y, 1 - ans).sum()
    #
    # # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(pd.crosstab(y, ans, rownames=['actuals'],
                      colnames=['predictions']))
    print("\n{:<11} {:.3f}".format('Recall:', recall))
    print("{:<11} {:.3f}".format('Precision:', precision))
    print("{:<11} {:.3f}".format('Accuracy:', accuracy))
    print()
