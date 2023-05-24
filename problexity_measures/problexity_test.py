import problexity as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# pre = 'C:/Users/alepa/Desktop/MGR/Code/'
# main_folder = 'results/amp2_wavdec/'

pre = 'C:/Users/alepa/Desktop/MGR/Code/'
main_folder = 'results/Barbara_13_05_2022_AB_wavdec/'

filenames = [
             ["features_WL_ZC_SSC.csv",[1,2]], # worse
             ["features_MAV.csv",[4,5]], # better
             ] 

strategy = 'ovo'

fig = plt.figure(figsize=(7,7))

for file in filenames:
    filename, classes = file
    print("\n\n")
    print(filename)
    dataset = pd.read_csv(pre+main_folder+filename, sep=",", decimal=".", header=None)
    dataset = dataset[dataset.iloc[:, -1].isin(classes)]
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values.astype(int)

    cc = px.ComplexityCalculator(multiclass_strategy=strategy)

    # Fit model with data
    cc.fit(X,y)
    print(f"Report: \n{cc.report()}\n")
    cc.plot(fig, (1,1,1))

    plt.tight_layout()
    plt.savefig(f"{pre}problexity_results/problexity_{strategy}_{','.join(map(str, classes))}_{filename.split('.')[0]}.png")


# # Begin problexity
# strategy = "ova"
# cc = px.ComplexityCalculator(multiclass_strategy=strategy)

# # Fit model with data
# cc.fit(X,y)
# print(f"Report: \n{cc.report()}\n")
# cc.plot(fig, (1,1,1))

# plt.tight_layout()
# plt.savefig(f"problexity_results/problexity_{strategy}_({','.join(map(str, class_combination))})_k={k}.png")

# # End problexity