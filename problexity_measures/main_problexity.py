import problexity as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest

def remove_k_worst_features(dataset, y, k):
    X = dataset.iloc[:, 0:-1].values
    X_new = SelectKBest()
    new_X = X_new.fit_transform(X, y)
    indexes_list = np.argpartition(X_new.scores_, k)
    worst_features_indexes = indexes_list[:k]
    worst_features_labels = dataset.columns[worst_features_indexes].to_list()
    X = np.delete(X, worst_features_indexes,1)
    return X, worst_features_labels

experiment_combinations = [
    # [<classes>, <k>],
    [[1,2,3,4,5], 3],
]

filename = "features_WL_ZC_VAR_MAV_SSC.csv"
main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/Barbara_wavdec/'

strategy = 'ova'

fig = plt.figure(figsize=(7,7))

for combination in experiment_combinations:
    classes, k = combination
    print("\n\n")
    print(filename)
    dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=0)
    dataset = dataset[dataset.iloc[:, -1].isin(classes)]
    y = dataset.iloc[:, -1].values.astype(int)

    X = remove_k_worst_features(dataset, y, k)

    # Begin problexity
    cc = px.ComplexityCalculator(multiclass_strategy=strategy)

    # Fit model with data
    cc.fit(X,y)
    print(f"Report: \n{cc.report()}\n")
    cc.plot(fig, (1,1,1))

    plt.tight_layout()
    plt.savefig(f"C:/Users/alepa/Desktop/MGR/problexity_results/problexity_{strategy}_({','.join(map(str, classes))})_k={k}.png")

    # End problexity