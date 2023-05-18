from itertools import combinations
import os
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from tabulate import tabulate
from scipy.stats import ttest_rel
print("")

main_folder = 'dataset_features/amp2_wavdec/'

filenames = os.listdir(main_folder)
classes = [1,2,3,4,5,6]
number_of_classes = len(classes)

list_combinations_classes = list()
for n in range(2, len(classes) + 1):
    list_combinations_classes += list(combinations(classes, n))
list_combinations_classes = list_combinations_classes[::-1] # reverse tuple

file_object = open(f'{main_folder}results_OvO_RFC_class_combinations.txt', 'w')
target_accuracy = 0.6

clfs = RandomForestClassifier(max_depth=2, random_state=11)

file_object = open('results_Select_K_Best', 'w')

scores = []
mean_scores = []

for k in range(0, 16):
    method_val = []
    mean_method_val = []
    for filename in filenames: 
        # file_object.write(f'k = {k}\n')
        dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=None)
        X = dataset.iloc[:, 0:-1].values
        y = dataset.iloc[:, -1].values.astype(int)

        # remove k worst features
        X_new = SelectKBest()
        new_X = X_new.fit_transform(X, y)
        indexes_list = np.argpartition(X_new.scores_, k)
        worst_features_indexes = indexes_list[:k]
        X = np.delete(X, worst_features_indexes,1)  
        # print(X.shape)

        kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
        splits = kfold.split(X,y)

        model = clfs['SVM']
        ovo = OneVsOneClassifier(model)

        mean_arr = []

        for n,(train_index,test_index) in enumerate(splits):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            ovo.fit(x_train_fold, y_train_fold)
            predict = ovo.predict(x_test_fold)
            mean_arr.append(metrics.accuracy_score(y_test_fold, predict))
        mean_method_val.append(round(np.mean(mean_arr),2))
        method_val.append(mean_arr)
    scores.append(method_val)

    file_object.write(f'{mean_method_val}\n')

alfa = .05
# print(f'test: {scores[0]}')

for k in range(0,16):
    file_object.write(f'\n\n############ k = {16-k}\n')
    score = scores[k]
    t_statistic = np.zeros((len(filenames), len(filenames)))
    p_value = np.zeros((len(filenames), len(filenames)))

    for i in range(len(filenames)):
        for j in range(len(filenames)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(score[i], score[j])
    # print("\n\nt-statistic for k = {k}\n", t_statistic, "\n\np-value:\n", p_value)
    # file_object.write(f'\n\nt-statistic for k = {k}\n{t_statistic}\n\np-value:\n {p_value}')

    headers = ["MAV", "SSC", "VAR", "WL", "ZC"]
    names_column = np.array([["MAV"], ["SSC"], ["VAR"], ["WL"], ["ZC"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    # file_object.write(f't-statistic:\n{t_statistic_table}\n\np-value:\n{p_value_table}')

    advantage = np.zeros((len(filenames), len(filenames)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\n\nAdvantage:\n", advantage_table)
    # file_object.write(f'\n\nAdvantage:\n{advantage_table}\n')

    significance = np.zeros((len(filenames), len(filenames)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print("Statistical significance (alpha = 0.05):\n", significance_table)
    # file_object.write(f'\n\nStatistical significance (alpha = 0.05):\n{significance_table}\n')

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    # print("Statistically significantly better:\n", stat_better_table)
    file_object.write(f'\n\nStatistically significantly better:\n{stat_better_table}\n')

file_object.close()