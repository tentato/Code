import numpy as np
import pandas as pd
import os
import pysiology.electromyography as electromyography

print(" ")

results_folder = 'results/Barbara_13_05_2022_AB/'
main_folder = 'C:/Users/alepa/Desktop/MGR/datasets/Barbara_13_05_2022_AB/'
classes_array = ['1', '2', '3', '4', '5']
# results_folder = 'results/KrzysztofJ_all/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/datasets/KrzysztofJ_all/'
# classes_array = ['1', '2', '3', '4', '5', '6','7', '8', '9']
# results_folder = 'results/MK/'
# main_folder = 'C:/Users/alepa/Desktop/MGR/datasets/MK_10_03_2022/'
# classes_array = ['1', '2', '3', '4', '5', '6','7', '8']

WL_table = []
ZC_table = []
VAR_table = []
MAV_table = []
SSC_table = []

for class_number in classes_array:
    for file_index, filename in enumerate(os.listdir(main_folder+class_number)):

        dataset = pd.read_csv(main_folder+class_number+'/'+filename, sep=";", decimal=",", header=None)

        WL_arr = np.zeros(17)
        ZC_arr = np.zeros(17).astype(int)
        VAR_arr = np.zeros(17)
        MAV_arr = np.zeros(17)
        SSC_arr = np.zeros(17).astype(int)

        for idx, col in enumerate(dataset.columns):
            column = dataset.loc[:, col]
            y = column.to_numpy()

            # Get the waveform length of the signal, a measure of complexity of the EMG Signal.
            WL_arr[idx] = electromyography.getWL(y)
            # How many times does the signal crosses the 0 (+-threshold).
            ZC_arr[idx] = electromyography.getZC(y, threshold=1)
            # Summation of average square values of the deviation of a variable.
            VAR_arr[idx] = electromyography.getVAR(y)
            # Thif functions compute the average of EMG signal Amplitude - Mean Absolute Value
            MAV_arr[idx] = electromyography.getMAV(y)
            # Number of times the slope of the EMG signal changes sign.
            SSC_arr[idx] = electromyography.getSSC(y, threshold=1)

        WL_arr[16] = int(class_number)
        ZC_arr[16] = int(class_number)
        VAR_arr[16] = int(class_number)
        MAV_arr[16] = int(class_number)
        SSC_arr[16] = int(class_number)

        WL_table.append(WL_arr)
        ZC_table.append(ZC_arr)
        VAR_table.append(VAR_arr)
        MAV_table.append(MAV_arr)
        SSC_table.append(SSC_arr)

    print("Class ",class_number," processed...")


WL_df = pd.DataFrame(data=WL_table)
ZC_df = pd.DataFrame(data=ZC_table)
VAR_df = pd.DataFrame(data=VAR_table)
MAV_df = pd.DataFrame(data=MAV_table)
SSC_df = pd.DataFrame(data=SSC_table)

folder_path = results_folder
os.makedirs(folder_path, exist_ok=True)  
WL_df.to_csv(folder_path+"features_WL.csv", header=False, index=False)
ZC_df.to_csv(folder_path+"features_ZC.csv", header=False, index=False)
VAR_df.to_csv(folder_path+"features_VAR.csv", header=False, index=False)
MAV_df.to_csv(folder_path+"features_MAV.csv", header=False, index=False)
SSC_df.to_csv(folder_path+"features_SSC.csv", header=False, index=False)