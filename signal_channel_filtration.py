import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import combinations
import pywt
from scipy.interpolate import interp1d

# This data set contains only 8 channels of EMG signals (no MMG data).
# This script performs wavelet transfor on all signals in dataset.

print(" ")
# print(pywt.wavelist(kind='discrete'))
# exit()

results_folder = 'C:/Users/alepa/Desktop/MGR/datasets/Barbara_13_05_2022_AB_wavdec/'
main_folder = 'C:/Users/alepa/Desktop/MGR/datasets/Barbara_13_05_2022_AB/'
classes_array = ['1', '2', '3', '4', '5']
os.makedirs(results_folder, exist_ok=True)  
for class_num in classes_array:
    os.makedirs(f"{results_folder}/{class_num}/", exist_ok=True)  


for class_number in classes_array:
    for file_index, filename in enumerate(os.listdir(main_folder+class_number)):
        dataset = pd.read_csv(main_folder+class_number+'/'+filename, sep=";", decimal=",", header=None)
        new_df = pd.DataFrame()
        new_df_arr = []
        i=0
        for (columnName, columnData) in dataset.items():
            # convert array to column
            coeffs = pywt.wavedec(columnData, 'db1', level=3)
            fig, ax = plt.subplots(4, 1, figsize=(10, 13))

            new_len = 1000
            f1 = interp1d(np.linspace(0, 1, len(coeffs[0])), coeffs[0], kind='linear')
            rescaled_arr1 = f1(np.linspace(0, 1, new_len))
            
            x = np.linspace(start = 0, stop = 1000, num = 1000) # for AB
            ax[0].plot(x, columnData)
            ax[1].plot(x, rescaled_arr1)
            # ax[1].plot(x, coeffs[1])
            # ax[2].plot(x, coeffs[2])
            # ax[3].plot(x, coeffs[3])
            ax[0].set_title("EMG")
            plt.tight_layout()
            plt.savefig("test.png")
            exit()
            new_arr = np.array(new_arr)
            new_df_arr.append(new_arr)
        new_df_arr = np.array(new_df_arr)
            
        print(new_df_arr.shape)
        exit()
        new_df.to_csv(f"{results_folder}/{class_num}/", header=False, index=False)
        
        
    print(f"Class {class_number} processed...")

print("\n\nFINISHED SUCCESSFULLY")