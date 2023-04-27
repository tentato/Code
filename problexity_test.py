import problexity as px
import matplotlib.pyplot as plt
import os
import pandas as pd


# WL - MAV, 1,4,5 
# 3 Classes, accuracy = 98%
# ZC - MAV - SSC, 1,3,4 
# 3 Classes, accuracy = 86%

main_folder = 'results/Barbara_13_05_2022_AB_wavdec/'
# main_folder = 'results/Barbara_13_05_2022_AB/'
# main_folder = 'results/KrzysztofJ_all/'
# main_folder = 'results/MK/'
# filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

# filename = "features_WL_MAV.csv"
filename = "features_ZC_MAV_SSC.csv"
# classes = [1,4,5]
classes = [1,3,4]
number_of_classes = len(classes)

fig = plt.figure(figsize=(7,7))


print("\n\n")
print(filename)
dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=None)
dataset = dataset[dataset.iloc[:, -1].isin(classes)]
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values.astype(int)

# print(y)
# exit()

# Initialize CoplexityCalculator with default parametrization
cc = px.ComplexityCalculator()

# Fit model with data
cc.fit(X,y)
print(f"Complexity{cc.complexity}\n")
print(f"Metrics{cc._metrics()}\n")
print(f"Score{cc.score()}\n")
print(f"Report{cc.report()}\n")
cc.plot(fig, (1,1,1))

plt.tight_layout()
plt.savefig(f"problexity_{filename.split('.')[0]}.png")