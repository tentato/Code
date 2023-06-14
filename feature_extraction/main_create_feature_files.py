import numpy as np
import pandas as pd
import os
import pysiology.electromyography as electromyography

# This data set contains 64 channels of EMG and MMG signals
# 16 channels * 4 decomposition levels = 64 channels
# the result should contain 64*5=320 features

# folder_path = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp3_wavdec/'
# main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2_wavdec/', 'C:/Users/alepa/Desktop/MGR/datasets/amp2_2_wavdec/']
# folder_path = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp2_2_wavdec/'
# main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2_2_wavdec/']
folder_path = 'C:/Users/alepa/Desktop/MGR/dataset_features/amp2_wavdec/'
main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/amp2_wavdec/']
classes_array = ['1', '2', '3', '4', '5', '6']

# folder_path = 'C:/Users/alepa/Desktop/MGR/dataset_features/Barbara_wavdec/'
# main_folder = ['C:/Users/alepa/Desktop/MGR/datasets/Barbara_wavdec/']
# classes_array = ['1', '2', '3', '4', '5']
os.makedirs(folder_path, exist_ok=True)  

WL_table = []
ZC_table = []
VAR_table = []
MAV_table = []
SSC_table = []
label = []

for class_number in classes_array:
    for fi, folder in enumerate(main_folder):
        for file_index, filename in enumerate(os.listdir(folder+class_number)):
            dataset = pd.read_csv(folder+class_number+'/'+filename, sep=",", decimal=".", header=None)
            rows_number, columns_number = dataset.shape

            WL_arr = np.zeros(columns_number)
            ZC_arr = np.zeros(columns_number)
            VAR_arr = np.zeros(columns_number)
            MAV_arr = np.zeros(columns_number)
            SSC_arr = np.zeros(columns_number)

            for idx, col in enumerate(dataset.columns):
                column = dataset.loc[:, col]
                y = column.to_numpy()

                # Get the waveform length of the signal, a measure of complexity of the EMG Signal.
                WL_arr[idx] = electromyography.getWL(y)
                # How many times does the signal crosses the 0 (+-threshold).
                ZC_arr[idx] = electromyography.getZC(y, threshold=0.01)
                # Summation of average square values of the deviation of a variable.
                VAR_arr[idx] = electromyography.getVAR(y)
                # Thif functions compute the average of EMG signal Amplitude - Mean Absolute Value
                MAV_arr[idx] = electromyography.getMAV(y)
                # Number of times the slope of the EMG signal changes sign.
                SSC_arr[idx] = electromyography.getSSC(y, threshold=0.01)

            WL_table.append(WL_arr)
            ZC_table.append(ZC_arr)
            VAR_table.append(VAR_arr)
            MAV_table.append(MAV_arr)
            SSC_table.append(SSC_arr)
            label.append(int(class_number))

    print(f"Class {class_number} processed...")

list_of_methods_names = ["WL", "ZC", "VAR", "MAV", "SSC"]

# create headers
headers_list = []
for method in list_of_methods_names:
    headers_list.append([(method+str(i)) for i in range(1, columns_number+1)])
headers_dict = dict(zip(list_of_methods_names, headers_list))

# create DataFrames
WL_df = pd.DataFrame(data=WL_table, columns=headers_dict["WL"])
WL_df = WL_df.astype(float)
ZC_df = pd.DataFrame(data=ZC_table, columns=headers_dict["ZC"])
ZC_df = ZC_df.astype(float)
VAR_df = pd.DataFrame(data=VAR_table, columns=headers_dict["VAR"])
VAR_df = VAR_df.astype(float)
MAV_df = pd.DataFrame(data=MAV_table, columns=headers_dict["MAV"])
MAV_df = MAV_df.astype(float)
SSC_df = pd.DataFrame(data=SSC_table, columns=headers_dict["SSC"])
SSC_df = SSC_df.astype(float)
label_df = pd.DataFrame(data=label, columns=["label"])

list_of_methods_tables = [WL_df, ZC_df, VAR_df, MAV_df, SSC_df]

comb_name = '_'.join(list_of_methods_names)
print(comb_name)
comb_df = pd.concat(list_of_methods_tables, axis=1)
comb_df = pd.concat([comb_df, label_df], axis=1)

comb_df.to_csv(f'{folder_path}features_{comb_name}.csv', header=True, index=False)

print("\n\nFINISHED SUCCESSFULLY")