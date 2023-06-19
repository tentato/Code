import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import floor, ceil

start_time = time.time()

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp3_wavdec/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp3.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp3 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_2_wavdec/'
# measures_filename = "C:/Users/alepa/Desktop/MGR/final results/problexity/amp2_2.txt"
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2_2 rfc.txt"

results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_wavdec/clsf_res/'
classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/Barbara_wavdec/clsf_res/'
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/barb rfc.txt"
os.makedirs(results_folder, exist_ok=True)  

clsf_ds = pd.read_csv(classification_filename, sep=";", decimal=".", header=0)

class_combinations = np.unique(np.array(clsf_ds["Class combination"].values))
numbers_of_classes = np.unique(np.array(clsf_ds["Number of classes"].values))

for number_of_classes in numbers_of_classes:
    temp_df = clsf_ds.copy()
    temp_df = temp_df[temp_df["Number of classes"] == number_of_classes]
    number_of_combinations = np.unique(np.array(temp_df["Class combination"].values))
    if "amp2" in results_folder:
        fig, ax = plt.subplots(int(ceil(len(number_of_combinations)/3)), 3, figsize=(15, int(ceil(len(number_of_combinations)/2))*3))
    else:
        fig, ax = plt.subplots(int(ceil(len(number_of_combinations)/2)), 2, figsize=(10, int(ceil(len(number_of_combinations)/2))*3))
    ax = ax.reshape(-1)
    for idx, class_combination in enumerate(number_of_combinations):
        print(idx)
        if len(class_combination) < 7:
            continue
        sub_clsf_ds = clsf_ds.copy()
        sub_clsf_ds = sub_clsf_ds[sub_clsf_ds["Class combination"] == class_combination]

        x = sub_clsf_ds["K worst features rejected"].values
        accuracy = sub_clsf_ds["Mean Accuracy"].values
        best_accuracies = np.sort(accuracy)[len(accuracy)-5:len(accuracy)]
        avg_best_accuracy = np.mean(best_accuracies)
        ax[idx].scatter(x, accuracy, s=3, c='blue', marker='o')
        ax[idx].set_ylabel('accuracy')
        ax[idx].set_xlabel('number of rejected worst features')
        ax[idx].set_title(f"{class_combination}, average_of_10_best={round(avg_best_accuracy, 3)}")

    plt.tight_layout()
    plt.savefig(f"{results_folder}{number_of_classes}.png")

    for i in range(0, len(ax)):
        ax[i].cla()
    print(f"{number_of_classes} processed...")

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")