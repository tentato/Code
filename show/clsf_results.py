import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import floor, ceil

start_time = time.time()

results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp3_wavdec/clsf_res/'
classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp3 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_2_wavdec/clsf_res/'
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2_2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/amp2_wavdec/clsf_res/'
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/amp2 rfc.txt"

# results_folder = 'C:/Users/alepa/Desktop/MGR/Code/show/Barbara_wavdec/clsf_res/'
# classification_filename = "C:/Users/alepa/Desktop/MGR/final results/barb rfc.txt"
os.makedirs(results_folder, exist_ok=True)  

clsf_ds = pd.read_csv(classification_filename, sep=";", decimal=".", header=0)

class_combinations = np.unique(np.array(clsf_ds["Class combination"].values))
numbers_of_classes = np.unique(np.array(clsf_ds["Number of classes"].values))

fig, ax = plt.subplots(1)
for idx, class_combination in enumerate(class_combinations):
    if len(class_combination) < 7:
        continue
    sub_clsf_ds = clsf_ds.copy()
    sub_clsf_ds = sub_clsf_ds[sub_clsf_ds["Class combination"] == class_combination]

    x = sub_clsf_ds["K worst features rejected"].values
    accuracy = sub_clsf_ds["Mean Accuracy"].values
    best_accuracies = np.sort(accuracy)[len(accuracy)-5:len(accuracy)]
    avg_best_accuracy = np.mean(best_accuracies)
    ax.scatter(x, accuracy, s=3, c='blue', marker='o')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('number of rejected worst features')
    ax.set_title(f"{class_combination}, average_of_10_best={round(avg_best_accuracy, 3)}")

    print(f"{class_combination} processed...")
    plt.tight_layout()
    plt.savefig(f"{results_folder}{class_combination}.png")

    ax.cla()

end_time = time.time()
print(f"Execution time: {round((end_time-start_time)/60,2)} minutes")