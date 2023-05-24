import time
import os
import numpy as np
import pandas as pd
import problexity as px
from itertools import combinations
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def min_max_normalize(dataset):
    X_df = dataset.iloc[:, 0:-1]
    y_df = dataset.iloc[:, -1]
    X_scaled = scaler.fit_transform(X_df)
    X_df = pd.DataFrame(X_scaled, columns=dataset.columns[:-1])
    dataset = pd.concat([X_df, y_df], axis=1)
    return dataset

def create_class_combinations(classes):
    list_combinations_classes = list()
    for n in range(2, len(classes) + 1):
        list_combinations_classes += list(combinations(classes, n))
    list_combinations_classes = list_combinations_classes[::-1] # reverse tuple
    return list_combinations_classes

start_time = time.time()

model = RandomForestClassifier(random_state=11) 
scaler = MinMaxScaler()

# main_folder = 'dataset_features/amp2_2_wavdec/'
main_folder = 'dataset_features/amp2_wavdec/'
# main_folder = 'dataset_features/Barbara_wavdec/'
filename = "features_WL_ZC_VAR_MAV_SSC.csv"
target_accuracy = 0.6

dataset = pd.read_csv(f"{main_folder}{filename}", sep=",", decimal=".", header=0)
dataset = min_max_normalize(dataset)
rows, columns = dataset.shape
features = columns - 1

classes = np.unique(np.array(dataset.iloc[:, -1].values))
number_of_classes = len(classes)

class_combinations = create_class_combinations(classes)

file_object = open(f'{main_folder}results_Select_K_Best.txt', 'w')
file_object.write(f'Class combination;Number of classes;K worst features rejected;Mean Accuracy;Worst features labels')  

scores = []
mean_scores = []

fig = plt.figure(figsize=(7,7))

for idx, class_combination in enumerate(class_combinations):
    for k in range(0, features):
        print(f"K: {k}")
        method_val = []
        mean_method_val = []

        subdataset = dataset[dataset.iloc[:, -1].isin(class_combination)]

        X = subdataset.iloc[:, 0:-1].values
        y = subdataset.iloc[:, -1].values.astype(int)

        # remove k worst features
        X_new = SelectKBest()
        new_X = X_new.fit_transform(X, y)
        indexes_list = np.argpartition(X_new.scores_, k)
        worst_features_indexes = indexes_list[:k]
        worst_features_labels = subdataset.columns[worst_features_indexes].to_list()
        X = np.delete(X, worst_features_indexes,1)  # after removing worst features

        # Begin problexity
        strategy = "ova"
        cc = px.ComplexityCalculator(multiclass_strategy=strategy)

        # Fit model with data
        cc.fit(X,y)
        print(f"Report: \n{cc.report()}\n")
        cc.plot(fig, (1,1,1))

        plt.tight_layout()
        plt.savefig(f"problexity_results/problexity_{strategy}_({','.join(map(str, class_combination))})_k={k}.png")

        # End problexity

        kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
        splits = kfold.split(X,y)

        ova = OneVsRestClassifier(model)

        balanced_accuraccy_array = []
        for n,(train_index,test_index) in enumerate(splits):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            ova.fit(x_train_fold, y_train_fold)
            predict = ova.predict(x_test_fold)

            ###Evaluating Prediction Accuracy
            if round(metrics.balanced_accuracy_score(y_test_fold, predict),2) < target_accuracy:
                print("RFC Acc: ",round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
                # file_object.write(f'\n{str(class_combination)};{len(class_combination)};{k};{target_accuracy};{" ".join(worst_features_labels)}')
                break
            print("RFC Acc: ",round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
            balanced_accuraccy_array.append(round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
        if len(balanced_accuraccy_array) > 0:
            file_object.write(f'\n{str(class_combination)};{len(class_combination)};{k};{round(np.mean(balanced_accuraccy_array),2)};{" ".join(worst_features_labels)}')  
    # break #just for first combination (all classes)

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")
file_object.close()