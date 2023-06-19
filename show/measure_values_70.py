import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy import spatial

start_time = time.time()

results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp3_wavdec/corr70/'
measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp3.txt"
classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp3 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_2_wavdec/corr70/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2_2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2_2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_wavdec/corr70/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/Barbara_wavdec/corr70/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/barb.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/barb rfc.txt"
os.makedirs(results_folder, exist_ok=True)  


measures_ds = pd.read_csv(measures_filename, sep=";", decimal=".", header=0)
clsf_ds = pd.read_csv(classification_filename, sep=";", decimal=".", header=0)

class_combinations = np.unique(np.array(clsf_ds["Class combination"].values))

measure_names = np.unique(np.array(measures_ds["Measure name"].values))

fig, ax = plt.subplots(9, 2, figsize=(10, 20))
ax = ax.reshape(-1)

file_object = open(f'{results_folder}correlation_values_70.txt', 'w')
# barb
# csv_header = 'Class combination;Number of classes;mean_good_accuracy;f1;f2;f3;f4;l1;l2;l3;n1;n2;n4;t4;sum_measures'
# amp2
# csv_header = 'Class combination;Number of classes;mean_good_accuracy;clsCoef;density;f1;f3;f4;l1;l2;l3;n1;n2;n4;sum_measures'
# amp2_2 i 3
csv_header = 'Class combination;Number of classes;mean_good_accuracy;clsCoef;density;f1;f2;f3;f4;hubs;l1;l2;l3;lsc;n1;n2;n4;t1;t2;t3;t4;sum_measures'



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
    best_accuracies_indices = np.argsort(accuracy)[-10:] # top 10 numbers
    best_accuracies = [accuracy[i] for i in best_accuracies_indices]
    avg_best_accuracy = np.mean(best_accuracies)

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
            best_scores = [scores[i] for i in best_accuracies_indices] ### znajdowanie najlepszych dodane
            avg_score = np.mean(best_scores)
            # ax[measure_idx].scatter(x, scores, s=3, c='red', marker='o')
            # ax[measure_idx].set_title(f"{measure_name}, best_avg={avg_score}")
            measure_correlation_score_pvalue.append(str(avg_score))
            measures_avg_scores.append(avg_score)
            print(f"{measure_name} - score: s={avg_score}")

    # plt.tight_layout()
    # plt.savefig(f"{results_folder}{class_combination}.png")

    for i in range(0, 18):
        ax[i].cla()
    print(f"{class_combination} processed...\n")

    file_object.write(f"\n{class_combination};{number_of_classes};{str(avg_best_accuracy).replace('.', ',')};{';'.join(measure_correlation_score_pvalue).replace('.', ',')};{str(np.sum(measures_avg_scores)).replace('.', ',')}")

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")