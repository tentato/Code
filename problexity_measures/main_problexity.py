import problexity as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

def min_max_normalize(dataset):
    scaler = MinMaxScaler()
    X_df = dataset.iloc[:, 0:-1]
    y_df = dataset.iloc[:, -1]
    X_scaled = scaler.fit_transform(X_df)
    X_df = pd.DataFrame(X_scaled, columns=dataset.columns[:-1])
    dataset = pd.concat([X_df, y_df], axis=1)
    return dataset

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
    [[1,2,3,4,5], 5],
    [[1,2,3,4,5], 10],
    [[1,2,3,4,5], 12],
    [[1,2,3,4,5], 25],
    [[1,2,3,4,5], 27],
    [[1,2,3,4,5], 60],
    [[1,2,3,4,5], 0],
    [[1,2,3,4,5], 3],
    [[1,2,3,4,5], 4],
    [[1,2,3,4,5], 7],
    [[1,2,3,4,5], 158],
    [[1,2,3,4,5], 159],
    [[1,2,3,4,5], 157],
    [[1,2,3,4,5], 156],
    [[1,2,3,4,5], 155],
    [[1,2,3,4,5], 154],
    [[1,2,3,4,5], 153],
    [[1,2,3,4,5], 151],
    [[1,2,3,4,5], 144],
    [[1,2,3,4,5], 146],
]

filename = "features_WL_ZC_VAR_MAV_SSC.csv"
main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/Barbara_wavdec/'
results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/Barbara_wavdec/'
file_object = open(f'{results_folder}results_problexity.txt', 'w')

strategy = 'ova'

fig = plt.figure(figsize=(7,7))

dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=0)
dataset = min_max_normalize(dataset)

for combination in experiment_combinations:
    classes, k = combination
    print(f"{classes} {k}\n")
    file_object.write(f"{classes} {k}\n")

    subdataset = dataset[dataset.iloc[:, -1].isin(classes)]
    y = subdataset.iloc[:, -1].values.astype(int)

    X, worst_features_labels = remove_k_worst_features(subdataset, y, k)

    cc = px.ComplexityCalculator(multiclass_strategy=strategy)
    cc.fit(X,y)
    report = cc.report()
    print(f"Complexities: {report['complexities']}")
    file_object.write(f"Complexities: {report['complexities']}\n\n")
    
    cc.plot(fig, (1,1,1))

    plt.tight_layout()
    plt.savefig(f"C:/Users/alepa/Desktop/MGR/problexity_results/problexity_{strategy}_({','.join(map(str, classes))})_k={k}.png")