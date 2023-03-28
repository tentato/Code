from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from itertools import combinations
print("")

main_folder = 'results/Barbara_13_05_2022_AB/'
# main_folder = 'results/KrzysztofJ_all/'
# main_folder = 'results/MK/'
# filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

filenames = os.listdir(main_folder)
classes = [1,2,3,4,5]
number_of_classes = len(classes)

list_combinations_classes = list()
for n in range(len(classes) + 1):
    list_combinations_classes += list(combinations(classes, n))
list_combinations_classes = list_combinations_classes[::-1] # reverse tuple

file_object = open(f'{main_folder}results_OvO_RFC_class_combinations.txt', 'w')

for filename in filenames:
    if "features_" in filename:
        print("\n\n")
        print(filename)
        file_object.write(f'\n\n{filename}')
        for idx, class_combination in enumerate(list_combinations_classes):
            dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=None)
            X = dataset.iloc[:, 0:-1].values
            y = dataset.iloc[:, -1].values.astype(int)
            kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
            splits = kfold.split(X,y)

            model = RandomForestClassifier(max_depth=2, random_state=11)
            # model = SVC()
            # model = GaussianNB()
            # model = KNeighborsClassifier() #best
            ovo = OneVsOneClassifier(model)

            balanced_accuraccy_array = []
            for n,(train_index,test_index) in enumerate(splits):
                x_train_fold, x_test_fold = X[train_index], X[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]
                ovo.fit(x_train_fold, y_train_fold)
                predict = ovo.predict(x_test_fold)
                ###Evaluating Prediction Accuracy
                file_object.write(f'\nRFC OvO Acc: {round(metrics.balanced_accuracy_score(y_test_fold, predict),2)}')
                print("RFC OvO Acc: ",round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
                balanced_accuraccy_array.append(round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
            file_object.write(f'\nRFC OvO Mean Acc: {round(np.mean(balanced_accuraccy_array),2)}')
