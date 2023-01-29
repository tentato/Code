import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pysiology.electromyography as electromyography

print(" ")

### This script prints EMG and MMG signal visualization for all sensors.
### One plot per one class.

results_folder = 'results/KrzysztofJ_all/'
main_folder = 'C:/Users/alepa/Desktop/MGR/datasets/KrzysztofJ_all/'
classes_array = ['1', '2', '3', '4', '5', '6','7', '8', '9']
FILES_NUMBER = 1

for class_number in classes_array:
    for file_index, filename in enumerate(os.listdir(main_folder+class_number)):

        dataset = pd.read_csv(main_folder+class_number+'/'+filename, sep=";", decimal=",", header=None)

        fig, ax = plt.subplots(8, 2, figsize=(10, 13))
        x = np.linspace(start = 0, stop = 3000, num = 3000)
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

            if idx % 2 == 0:
                ax[int(idx/2), 0].plot(x, y)
                ax[0,0].set_title("MMG")
            else:
                ax[int(idx/2), 1].plot(x, y)
                ax[0,1].set_title("EMG")

        plt.tight_layout()
        plt.savefig("{}{}_{}_results.png".format(results_folder, class_number, filename.split('.')[0]))
        print("{}{}_{}_results.png saved...".format(results_folder, class_number, filename.split('.')[0]))

        if file_index >= FILES_NUMBER-1:
            break