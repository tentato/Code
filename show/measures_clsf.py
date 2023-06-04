from itertools import combinations
import time
import problexity as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
import problexity.classification as pc

start_time = time.time()

measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/barb.txt"
classification_filename = "C:/Users/alepa/Desktop/MGR/final results/barb rfc.txt"
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp3_wavdec/'
# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/amp3_wavdec/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp2_2_wavdec/'
# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/amp2_2_wavdec/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp2_wavdec/'
# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/problexity_measures/amp2_wavdec/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/dataset_features/Barbara_wavdec/'
results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/Barbara_wavdec/'

measures_ds = pd.read_csv(measures_filename, sep=";", decimal=".", header=0)
clsf_ds = pd.read_csv(classification_filename, sep=";", decimal=".", header=0)

class_combinations = np.unique(np.array(clsf_ds["Class combination"].values))
measure_names = np.unique(np.array(measures_ds["Measure name"].values))

fig, ax = plt.subplots(10, 2, figsize=(10, 20))
ax = ax.reshape(-1)

for idx, class_combination in enumerate(class_combinations):
    sub_clsf_ds = clsf_ds.copy()
    sub_clsf_ds = sub_clsf_ds[sub_clsf_ds["Class combination"] == class_combination]
    rows, columns = sub_clsf_ds.shape

    x = sub_clsf_ds["K worst features rejected"].values
    accuracy = sub_clsf_ds["Mean Accuracy"].values
    max_accuracy = np.sort(accuracy)[-1]
    ax[0].scatter(x, accuracy, s=3, c='red', marker='o')
    ax[0].set_title(f"balanced_accuracy, best={max_accuracy}")

    sub_meas_ds = measures_ds.copy()
    sub_meas_ds = sub_meas_ds[sub_meas_ds["Class combination"] == class_combination]
    for measure_idx, measure_name in enumerate(measure_names):
        sub_meas_single_ds = sub_meas_ds.copy()
        sub_meas_single_ds = sub_meas_single_ds[sub_meas_single_ds["Measure name"] == measure_name]
        scores = sub_meas_single_ds["Measure score"].values
        ax[measure_idx+1].scatter(x, scores, s=3, c='red', marker='o')
        ax[measure_idx+1].set_title(measure_name)

    plt.tight_layout()
    plt.savefig(f"{results_folder}{class_combination}.png")

    for i in range(0, 20):
        ax[i].cla()

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")