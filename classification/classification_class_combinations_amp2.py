from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from itertools import combinations
print("")

main_folder = 'dataset_features/amp2_wavdec/'

filenames = os.listdir(main_folder)
classes = [1,2,3,4,5,6]
# classes = ['1','2','3','4','5','6']
number_of_classes = len(classes)

list_combinations_classes = list()
for n in range(2, len(classes) + 1):
    list_combinations_classes += list(combinations(classes, n))
list_combinations_classes = list_combinations_classes[::-1] # reverse tuple

file_object = open(f'{main_folder}results_OvO_RFC_class_combinations.txt', 'w')
target_accuracy = 0.6

for filename in filenames:
    if "features_" in filename:
        print("\n\n")
        print(filename)
        for idx, class_combination in enumerate(list_combinations_classes):
            dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=0)
            dataset = dataset[dataset.iloc[:, -1].isin(class_combination)]
            X = dataset.iloc[:, 0:-1].values
            y = dataset.iloc[:, -1].values.astype(int)
            kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
            splits = kfold.split(X,y)

            model = RandomForestClassifier(max_depth=2, random_state=11) # the best
            ovr = OneVsRestClassifier(model)

            balanced_accuraccy_array = []
            for n,(train_index,test_index) in enumerate(splits):
                x_train_fold, x_test_fold = X[train_index], X[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]
                ovr.fit(x_train_fold, y_train_fold)
                predict = ovr.predict(x_test_fold)

                ###Evaluating Prediction Accuracy
                if round(metrics.balanced_accuracy_score(y_test_fold, predict),2) < target_accuracy:
                    file_object.write(f'\nFile {filename} class combination {str(class_combination)}. Accuracy lower than {target_accuracy}, skipping...')
                    break
                print("RFC Acc: ",round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
                balanced_accuraccy_array.append(round(metrics.balanced_accuracy_score(y_test_fold, predict),2))
            if len(balanced_accuraccy_array) > 0:
                file_object.write(f'\nFile {filename} class combination {str(class_combination)} Mean Accuracy: {round(np.mean(balanced_accuraccy_array),2)}')

file_object.close()