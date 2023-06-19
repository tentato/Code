import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy import spatial

start_time = time.time()

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp3_wavdec/corr90/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp3.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp3 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_2_wavdec/corr90/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2_2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2_2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_wavdec/corr90/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2 rfc.txt"

results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/Barbara_wavdec/corr90/'
measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/barb.txt"
classification_filename = "C:/Users/alepa/Desktop/MGR/final results/barb rfc.txt"
os.makedirs(results_folder, exist_ok=True)  

measures_ds = pd.read_csv(measures_filename, sep=";", decimal=".", header=0)
clsf_ds = pd.read_csv(classification_filename, sep=";", decimal=".", header=0)

class_combinations = np.unique(np.array(clsf_ds["Class combination"].values))

measure_names = np.unique(np.array(measures_ds["Measure name"].values))

fig, ax = plt.subplots(10, 2, figsize=(10, 20))
ax = ax.reshape(-1)

file_object = open(f'{results_folder}correlation_values_90.txt', 'w')
# barb
csv_header = 'Class combination;Number of classes;mean_good_accuracy;l2;l3;n1;n4;sum_measures'
# amp2
# csv_header = 'Class combination;Number of classes;mean_good_accuracy;l1;l2;l3;n1;sum_measures'

file_object.write(csv_header)  

for idx, class_combination in enumerate(class_combinations):
    if len(class_combination) < 7:
        continue
    sub_clsf_ds = clsf_ds.copy()
    sub_clsf_ds = sub_clsf_ds[sub_clsf_ds["Class combination"] == class_combination]
    rows, columns = sub_clsf_ds.shape
    number_of_classes = np.array(sub_clsf_ds["Number of classes"].values)[0]

    x = sub_clsf_ds["K worst features rejected"].values
    accuracy = sub_clsf_ds["Mean Accuracy"].values
    max_accuracy = np.max(accuracy)
    min_accuracy = np.min(accuracy)
    best_accuracies = np.sort(accuracy)[len(accuracy)-5:len(accuracy)]
    avg_best_accuracy = np.mean(best_accuracies)
    ax[0].scatter(x, accuracy, s=3, c='red', marker='o')
    ax[0].set_title(f"balanced_accuracy, best_avg={round(avg_best_accuracy, 3)}")

    sub_meas_ds = measures_ds.copy()
    sub_meas_ds = sub_meas_ds[sub_meas_ds["Class combination"] == class_combination]
    measure_correlation_score_pvalue = []
    measures_avg_scores = []
    for measure_idx, measure_name in enumerate(measure_names):
        if measure_name in csv_header:
            sub_meas_single_ds = sub_meas_ds.copy()
            sub_meas_single_ds = sub_meas_single_ds[sub_meas_single_ds["Measure name"] == measure_name]
            scores = sub_meas_single_ds["Measure score"].values
            if measure_name in 'clsCoef;density;hubs;lsc;n2;t1;t2;t3':
                scores = 1 - scores # revert
            avg_score = np.mean(np.sort(scores)[0:5]) # DODAĆ ODNAJDOWANIE INDEKSÓW
            ax[measure_idx+1].scatter(x, scores, s=3, c='red', marker='o')
            ax[measure_idx+1].set_title(f"{measure_name}, best_avg={avg_score}")
            measure_correlation_score_pvalue.append(str(avg_score))
            measures_avg_scores.append(avg_score)

    # plt.tight_layout()
    # plt.savefig(f"{results_folder}{class_combination}.png")

    for i in range(0, 20):
        ax[i].cla()
    print(f"{class_combination} processed...\n")

    file_object.write(f"\n{class_combination};{number_of_classes};{str(avg_best_accuracy).replace('.', ',')};{';'.join(measure_correlation_score_pvalue).replace('.', ',')};{str(round(np.sum(measures_avg_scores), 3)).replace('.', ',')}")

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")