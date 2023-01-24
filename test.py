import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pysiology.electromyography as electromyography

print(" ")

# Kolumny odpowiadają kolejnym kanałom zarejestrowanego sygnału. 
#   	Kolumny o numerach parzystych (0,2,4) są związane z sygnałami __MMG__.
#   	Kolumny o numerach nieparzystych (1,3,5) są związane z sygnałami __EMG__.
#   	Kanałów jest 16 -- 8 MMG i 8 EMG
# Kolejne wiersze pliku __.csv__ reprezentują kolejne próbki sygnału. 

results_folder = 'results/'
main_folder = 'C:/Users/alepa/Desktop/MGR/ProfKurzynski/KrzysztofJ_all/KrzysztofJ_all/'
# classes_array = ['1']
classes_array = ['2', '3', '4', '5', '6','7', '8', '9']
# FILES_NUMBER = 1

for class_number in classes_array:
    for file_index, filename in enumerate(os.listdir(main_folder+class_number)):

        dataset = pd.read_csv(main_folder+class_number+'/'+filename, sep=";", decimal=",", header=None)

        # fig, ax = plt.subplots(8, 2, figsize=(10, 13))
        # x = np.linspace(start = 0, stop = 3000, num = 3000)
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

            # if idx % 2 == 0:
            #     ax[int(idx/2), 0].plot(x, y)
            #     ax[0,0].set_title("MMG")
            # else:
            #     ax[int(idx/2), 1].plot(x, y)
            #     ax[0,1].set_title("EMG")

        # plt.tight_layout()
        # plt.savefig("{}{}_{}_results.png".format(results_folder, class_number, filename.split('.')[0]))
        # print("{}{}_{}_results.png saved...".format(results_folder, class_number, filename.split('.')[0]))
        # print("WL array:\n",WL_arr)
        # print("ZC array:\n",ZC_arr)
        # print("VAR array:\n",VAR_arr)
        # print("MAV array:\n",MAV_arr)
        # print("SSC array:\n",SSC_arr)

        result_df = pd.DataFrame(data=[WL_arr, ZC_arr, VAR_arr, MAV_arr, SSC_arr],
                                index=["WL_arr", "ZC_arr", "VAR_arr", "MAV_arr", "SSC_arr"],
                                columns=dataset.columns)
        folder_path = results_folder+class_number+"/"
        os.makedirs(folder_path, exist_ok=True)  
        result_df.to_csv(folder_path+filename.split(".")[0]+"_features_1.csv", header=False) 

        # print("\nRESULT TABLE:\n", result_df)

        # if file_index >= FILES_NUMBER-1:
        #     break