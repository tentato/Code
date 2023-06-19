import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import combinations
import pywt
from scipy.interpolate import interp1d

###### NOT READY

# results_folder = 'C:/Users/alepa/Desktop/MGR/datasets/amp2_2_first_20_wavdec/'
# main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2_2_first_20/']

# results_folder = 'C:/Users/alepa/Desktop/MGR/datasets/amp2_2_wavdec/'
# main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2_2/']

main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2/test1/', 'C:/Users/alepa/Desktop/MGR/datasets/amp2/test2/']
results_folder = 'C:/Users/alepa/Desktop/MGR/datasets/amp2_wavdec/'
classes_array = ['1', '2', '3', '4', '5', '6']

# results_folder = 'C:/Users/alepa/Desktop/MGR/datasets/Barbara_wavdec/'
# main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/Barbara/']
# classes_array = ['1', '2', '3', '4', '5']

os.makedirs(results_folder, exist_ok=True)  
for class_num in classes_array:
    os.makedirs(f"{results_folder}/{class_num}/", exist_ok=True)  

for class_number in classes_array:
    print(f"Processing files for class {class_number}")
    for fi, folder in enumerate(main_folder):
        for file_index, filename in enumerate(os.listdir(folder+class_number)):
            if ".csv" not in filename:
                continue
            print(f"Processing {filename} file...")
            dataset = pd.read_csv(folder+class_number+'/'+filename, sep=";", decimal=",", header=None)
            if "amp2" in folder:
                dataset = dataset.iloc[:, 0:16]
                channel_indexes = [dataset.columns.get_loc(c) for c in dataset.columns if c in dataset]
                MMG_channels_indexes = [channel_index for channel_index in channel_indexes if channel_index % 2 == 0]
                dataset = dataset.drop(MMG_channels_indexes, axis=1)
            
            rows_number, columns_number = dataset.shape
            new_df_arr = []
            i=0
            for (columnName, columnData) in dataset.items():
                coeffs = pywt.wavedec(columnData, 'db1', level=3)
                f1 = interp1d(np.linspace(0, 1, len(coeffs[0])), coeffs[0], kind='linear')
                rescaled_arr1 = f1(np.linspace(0, 1, rows_number))
                f2 = interp1d(np.linspace(0, 1, len(coeffs[1])), coeffs[1], kind='linear')
                rescaled_arr2 = f2(np.linspace(0, 1, rows_number))
                f3 = interp1d(np.linspace(0, 1, len(coeffs[2])), coeffs[2], kind='linear')
                rescaled_arr3 = f3(np.linspace(0, 1, rows_number))
                f4 = interp1d(np.linspace(0, 1, len(coeffs[3])), coeffs[3], kind='linear')
                rescaled_arr4 = f4(np.linspace(0, 1, rows_number))

                print(len(coeffs[0]))
                print(len(coeffs[1]))
                print(len(coeffs[2]))
                print(len(coeffs[3]))
                
                #### PRINTING SIGNALS #####
                fig, ax = plt.subplots(5, 1, figsize=(10, 10))
                x = np.linspace(start = 0, stop = 2000, num = 2000) # for AB
                ax[0].plot(x, columnData)
                ax[0].set_title("EMG")
                ax[0].set_ylabel("signal")
                ax[1].plot(x, rescaled_arr1)
                ax[1].set_ylabel("A3", rotation = 90)
                # ax[1].set_title("0Hz - 125Hz")
                ax[2].plot(x, rescaled_arr2)
                # ax[2].set_title("125Hz - 250Hz")
                ax[2].set_ylabel("D3")
                ax[3].plot(x, rescaled_arr3)
                ax[3].set_ylabel("D2")
                # ax[3].set_title("250Hz - 500Hz")
                ax[4].plot(x, rescaled_arr4)
                ax[4].set_ylabel("D1")
                # ax[4].set_title("500Hz - 1000Hz")
                plt.tight_layout()
                plt.savefig("dec_visualization.png")

                new_df_arr.append(rescaled_arr1)
                new_df_arr.append(rescaled_arr2)
                new_df_arr.append(rescaled_arr3)
                new_df_arr.append(rescaled_arr4)
                exit()
            
            new_df_arr = np.array(new_df_arr)
            new_df_arr = np.transpose(new_df_arr)
            new_df = pd.DataFrame(new_df_arr)

            print(f"Saving decomposition results as {results_folder}{class_number}/{fi}{filename}")
            new_df.to_csv(f"{results_folder}{class_number}/{fi}{filename}", header=False, index=False)
        
        
    print(f"Class {class_number} processed...")

print("\n\nFINISHED SUCCESSFULLY")