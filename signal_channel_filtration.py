import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pysiology.electromyography as electromyography
from itertools import combinations

# This data set contains only 8 channels of EMG signals (no MMG data)

print(" ")

results_folder = 'results/Barbara_13_05_2022_AB/'
main_folder = 'C:/Users/alepa/Desktop/MGR/datasets/Barbara_13_05_2022_AB/'
folder_path = results_folder
os.makedirs(folder_path, exist_ok=True)  
classes_array = ['1', '2', '3', '4', '5']


for class_number in classes_array:
    for file_index, filename in enumerate(os.listdir(main_folder+class_number)):
        dataset = pd.read_csv(main_folder+class_number+'/'+filename, sep=";", decimal=",", header=None)
        new_df = pd.DataFrame()
        i=0
        for (columnName, columnData) in dataset.iteritems():
            new_df[str(i)] = columnData
            new_df[str(i+1)] = columnData
            i+=2
        
    print(f"Class {class_number} processed...")

print("\n\nFINISHED SUCCESSFULLY")