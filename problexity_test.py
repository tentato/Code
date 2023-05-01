import problexity as px
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


main_folder = 'results/Barbara_13_05_2022_AB_wavdec/'
# main_folder = 'results/Barbara_13_05_2022_AB/'
# main_folder = 'results/KrzysztofJ_all/'
# main_folder = 'results/MK/'
# filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

###
# filename = "features_WL_MAV.csv"
# filename = "features_WL_ZC_MAV.csv"
# filename = "features_WL_ZC_MAV_SSC.csv"
# classes = [1,4,5]
###

filename = "features_ZC_MAV_SSC.csv"
classes = [1,3,4]
number_of_classes = len(classes)

fig = plt.figure(figsize=(7,7))


print("\n\n")
print(filename)
dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=None)
dataset = dataset[dataset.iloc[:, -1].isin(classes)]
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values.astype(int)


cc = px.ComplexityCalculator(multiclass_strategy='ova')

def f1(X, y):
    """
    Calculates the Maximum Fisher's discriminant ratio (F1) metric.
    Measure describes the overlap of feature values in each class.
    .. math::
        F1=\\frac{1}{1+max^{m}_{i=1}r_{f_{i}}}
    :type X: array-like, shape (n_samples, n_features)
    :param X: Dataset
    :type y: array-like, shape (n_samples)
    :param y: Labels
    :rtype: float
    :returns: F1 score
    """
    
    X = np.copy(X)
    y = np.copy(y)

    X_0 = X[y==0]
    X_1 = X[y==1]

    try:
        X_0_prop = X_0.shape[0]/X.shape[0]
        X_1_prop = X_1.shape[0]/X.shape[0]
    except:
        return np.nan

    X_0_mean = np.mean(X_0, axis = 0)
    X_1_mean = np.mean(X_1, axis = 0)

    X_0_std = np.std(X_0, axis = 0)
    X_1_std = np.std(X_1, axis = 0)

    l = (X_0_prop*X_1_prop*np.power((X_0_mean - X_1_mean),2)) + (X_1_prop*X_0_prop*np.power((X_1_mean - X_0_mean),2))
    m = (X_0_prop*(np.power(X_0_std,2))) + (X_1_prop*(np.power(X_1_std,2)))
    # print(l)
    # print(m)
    r_all = l/m
    print(f"R_ALL: {r_all}")
    print(1 / (1+np.max(r_all)))
    return 1 / (1+np.max(r_all))

complexity = []

for c in classes:
    complexity.append(f1(X, (y == c).astype(int)))

print(complexity)
exit()
# Fit model with data
# cc.fit(X,y)
# print(f"Report: \n{cc.report()}\n")
# cc.plot(fig, (1,1,1))

# plt.tight_layout()
# plt.savefig(f"problexity_{filename.split('.')[0]}.png")