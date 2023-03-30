import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pysiology.electromyography as electromyography
from itertools import combinations
import pywt

# This data set contains only 8 channels of EMG signals (no MMG data)

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
        for (columnName, columnData) in dataset.iteritems():
            # new_df[str(i)] = columnData
            # new_df[str(i+1)] = columnData
            # i+=2
            # convert array to column

            new_arr = pywt.wavedec(columnData, 'coif1', level=3)
            print(len(new_arr[1]))
            exit()
            new_arr = np.array(new_arr)
            new_df_arr.append(new_arr)
        new_df_arr = np.array(new_df_arr)
            
        print(new_df_arr.shape)
        exit()
        new_df.to_csv(f"{results_folder}/{class_num}/", header=False, index=False)
        
        
    print(f"Class {class_number} processed...")

print("\n\nFINISHED SUCCESSFULLY")