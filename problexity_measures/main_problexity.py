from itertools import combinations
import time
import problexity as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
import problexity.classification as pc

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

def create_class_combinations(classes):
    list_combinations_classes = list()
    for n in range(3, len(classes) + 1):
        list_combinations_classes += list(combinations(classes, n))
    list_combinations_classes = list_combinations_classes[::-1] # reverse tuple
    return list_combinations_classes

start_time = time.time()

filename = "features_WL_ZC_VAR_MAV_SSC.csv"
main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp3_wavdec/'
results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/amp3_wavdec/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp2_2_wavdec/'
# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/amp2_2_wavdec/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp2_wavdec/'
# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/amp2_wavdec/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/Barbara_wavdec/'
# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/Barbara_wavdec/'

file_object = open(f'{results_folder}results_problexity.txt', 'w')
file_object.write(f'Class combination;Number of classes;K worst features rejected;Measure name;Measure score')  

dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=0)
dataset = min_max_normalize(dataset)

rows, columns = dataset.shape
features = columns - 1

classes = np.unique(np.array(dataset.iloc[:, -1].values))
class_combinations = create_class_combinations(classes)
metrics_array = [pc.f1, pc.f2, pc.f3, pc.f4, pc.l1, pc.l2, pc.l3, pc.n1, pc.n2, pc.n4, pc.t1, pc.lsc, pc.t2, pc.t3, pc.t4, pc.density, pc.clsCoef, pc.hubs]

for idx, class_combination in enumerate(class_combinations):
    number_of_classes = len(class_combination)
    for metric in metrics_array:
        for k in range(0, features):
            method_val = []
            mean_method_val = []
            subdataset = dataset.copy()
            subdataset = subdataset[subdataset.iloc[:, -1].isin(class_combination)]
            y = subdataset.iloc[:, -1].values.astype(int)

            X, worst_features_labels = remove_k_worst_features(subdataset, y, k)
            cc = px.ComplexityCalculator(metrics=[metric], colors=['#FD0100'], ranges={'LI': 1}, weights=np.ones((1)), mode='classification', multiclass_strategy='ova')
            cc.fit(X,y)
            report = cc.report()
            measure, measure_score = list(report["complexities"].items())[0]
            file_object.write(f'\n{str(class_combination)};{len(class_combination)};{k};{measure};{measure_score}')  
            print(f'{str(class_combination)};{len(class_combination)};{k};{measure};{measure_score}')

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")
file_object.close()