import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy import spatial

start_time = time.time()

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp3_wavdec/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp3.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp3 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_2_wavdec/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2_2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2_2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_wavdec/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2 rfc.txt"

results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/Barbara_wavdec/'
measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/barb.txt"
classification_filename = "C:/Users/alepa/Desktop/MGR/final results/barb rfc.txt"

measures_ds = pd.read_csv(measures_filename, sep=";", decimal=".", header=0)
clsf_ds = pd.read_csv(classification_filename, sep=";", decimal=".", header=0)

class_combinations = np.unique(np.array(clsf_ds["Class combination"].values))

measure_names = np.unique(np.array(measures_ds["Measure name"].values))

fig, ax = plt.subplots(10, 2, figsize=(10, 20))
ax = ax.reshape(-1)

file_object = open(f'{results_folder}correlation.txt', 'w')
# file_object = open(f'{results_folder}spearman_correlation.txt', 'w')
# csv_header = 'Class combination;Number of classes;mean_good_accuracy;f1;min_f1;f2;min_f2;f3;min_f3;f4;min_f4;l1;min_l1;l2;min_l2;l3;min_l3;n1;min_n1;n2;min_n2;n4;min_n4'
csv_header = 'Class combination;Number of classes;mean_good_accuracy;l1;min_l1;l2;min_l2;l3;min_l3;n1;min_n1;n2;min_n2;n4;min_n4;sum_measures;mean_measures'
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
    best_accuracies = np.sort(accuracy[0:int(len(accuracy)*0.1)])
    avg_best_accuracy = np.mean(np.sort(accuracy[0:int(len(accuracy)*0.1)])) # 20%
    ax[0].scatter(x, accuracy, s=3, c='red', marker='o')
    ax[0].set_title(f"balanced_accuracy, best_avg={avg_best_accuracy}")

    sub_meas_ds = measures_ds.copy()
    sub_meas_ds = sub_meas_ds[sub_meas_ds["Class combination"] == class_combination]
    measure_correlation_score_pvalue = []
    measures_avg_scores = []
    for measure_idx, measure_name in enumerate(measure_names):
        if measure_name in csv_header:
            sub_meas_single_ds = sub_meas_ds.copy()
            sub_meas_single_ds = sub_meas_single_ds[sub_meas_single_ds["Measure name"] == measure_name]
            scores = sub_meas_single_ds["Measure score"].values
            max_score = np.max(scores)
            min_score = np.min(scores)
            avg_score = np.mean(np.sort(scores[0:int(len(scores)*0.1)])) # 20% DODAĆ ODNAJDOWANIE INDEKSÓW
            ax[measure_idx+1].scatter(x, scores, s=3, c='red', marker='o')
            ax[measure_idx+1].set_title(f"{measure_name}, best_avg={avg_score}")
            correlation = abs(np.corrcoef(accuracy, scores)[0, 1])
            measure_correlation_score_pvalue.append(str(round(correlation, 3)))
            measure_correlation_score_pvalue.append(str(round(avg_score, 3)))
            measures_avg_scores.append(avg_score)
            print(f"{measure_name} - Correlation: s={round(correlation, 3)}")
            print(f"{measure_name} - score: s={round(avg_score, 3)}")

    plt.tight_layout()
    plt.savefig(f"{results_folder}{class_combination}.png")

    for i in range(0, 20):
        ax[i].cla()
    print(f"{class_combination} processed...\n")

    file_object.write(f"\n{class_combination};{number_of_classes};{str(avg_best_accuracy).replace('.', ',')};{';'.join(measure_correlation_score_pvalue).replace('.', ',')};{str(round(np.sum(measures_avg_scores), 3)).replace('.', ',')};{str(round(np.mean(measures_avg_scores), 3)).replace('.', ',')}")

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")