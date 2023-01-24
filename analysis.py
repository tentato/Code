import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
print("")

filename = "features_MAV.csv"
#filenames =["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

print("")
dataset = pd.read_csv("Code/results/"+filename, sep=",", decimal=".", header=None)
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values.astype(int)

n_features = X.shape[1]
print(n_features)

fig, axs = plt.subplots(1,1, figsize=(8,8))
X_t = PCA(n_components=2).fit_transform(X)
print(X_t.shape)
print(y.shape)

axs.scatter(*X_t.T, c=y)

plt.tight_layout()
plt.savefig("PCA.png")

fig, axs = plt.subplots(n_features,n_features, figsize=(100,100))

for i in range(n_features):
    for j in range(n_features):
        if i==j:
            continue
        axs[i,j].scatter(X[:,i], X[:,j], c=y)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.tight_layout()
plt.savefig("fig.png")

