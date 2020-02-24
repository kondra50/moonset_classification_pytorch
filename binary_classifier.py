import sklearn.datasets
import matplotlib.pyplot as plt
import torch

X,y = sklearn.datasets.make_moons(500,noise=0.1)
plt.scatter(X[:,0],X[:,1],s=40,c=y)
plt.show()
plt.savefig('myfig.png' )

# numpy to tensor
X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)