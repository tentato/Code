import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pysiology.electromyography as electromyography

print(" ")

### This script prints EMG signal visualization for all channels.
### One plot per one class.


results_folder = 'results/amp2/'
main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2/test1/', 'C:/Users/alepa/Desktop/MGR/datasets/amp2/test2/']
classes_array = ['1', '2', '3', '4', '5', '6']
FILES_NUMBER = 1

for class_number in classes_array:
    for file_index, filename in enumerate(os.listdir(main_folder[0]+class_number)):

        dataset = pd.read_csv(main_folder[0]+class_number+'/'+filename, sep=";", decimal=",", header=None)
        dataset = dataset.iloc[:, 0:16]

        fig, ax = plt.subplots(8, 2, figsize=(10, 13))
        x = np.linspace(start = 0, stop = 2000, num = 2000)
        WL_arr = np.zeros(16)
        ZC_arr = np.zeros(16).astype(int)
        VAR_arr = np.zeros(16)
        MAV_arr = np.zeros(16)
        SSC_arr = np.zeros(16).astype(int)

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

            ax[idx].plot(x, y)
            ax[0].set_title("EMG")

        plt.tight_layout()
        plt.savefig("{}{}_{}_results.png".format(results_folder, class_number, filename.split('.')[0]))
        print("{}{}_{}_results.png saved...".format(results_folder, class_number, filename.split('.')[0]))

        if file_index >= FILES_NUMBER-1:
            break