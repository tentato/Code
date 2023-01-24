from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest
print("")
matplotlib.style.use('ggplot')

# filenames = ["features_WL.csv"]
filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

for filename in filenames:
    dataset = pd.read_csv("Code/results/"+filename, sep=",", decimal=".", header=None, 
        names=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16"])
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values.astype(int)

    class_pairs = []
    class_numbers = [*set(y)] # removes duplcates
    for i, a in enumerate(class_numbers):
        for j, b in enumerate(class_numbers):
            if a==b:
                break
            class_pairs.append([a,b])

    # LDA all
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    lda = LDA(n_components=2)
    X_l = lda.fit(X, y).transform(X)
    ax.scatter(*X_l.T, c=y, cmap="hsv")
    plt.tight_layout()
    plt.savefig("LDA/"+filename.split(".")[0]+"_LDA.png")

    # PCA all
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    X_p = PCA(n_components=2).fit_transform(X)
    ax.scatter(*X_p.T, c=y, cmap="hsv")
    plt.tight_layout()
    plt.savefig("PCA/"+filename.split(".")[0]+"_PCA.png")


    ### Original
    fig, ax = plt.subplots(9, 9, figsize=(100,100))
    for pair in class_pairs:
        print(pair)
        filtered_df = dataset[(dataset['c16'] == pair[0]) | (dataset['c16'] == pair[1])] 
        X_f = filtered_df.iloc[:, 0:-1].values
        y_f = filtered_df.iloc[:, -1].values.astype(int)
        # PCA
        X_p = PCA(n_components=2).fit_transform(X_f)
        ax[pair[0]-1, pair[1]-1].scatter(*X_p.T, c=y_f, cmap="cool")
        # LDA
        X_l = LDA().fit_transform(X_f, y_f)
        ax[pair[1]-1, pair[0]-1].scatter(X_l, X_l, c=y_f, cmap="cool")

        if pair[1]-1 == 0:
            ax[pair[1]-1, pair[0]-1].set_title(pair[0])
        if pair[1]-1 == 0:
            ax[pair[0]-1, pair[1]-1].set_ylabel(pair[0])

        if pair[1]-1 == 0:
            ax[pair[1]-1, pair[0]-1].set_title(pair[0])
        if pair[1]-1 == 0:
            ax[pair[0]-1, pair[1]-1].set_ylabel(pair[0])

        kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
        splits = kfold.split(X_l,y_f)

        GNB = GaussianNB()

        for n,(train_index,test_index) in enumerate(splits):
            X_prain_fold, X_pest_fold = X_l[train_index], X_l[test_index]
            y_train_fold, y_test_fold = y_f[train_index], y_f[test_index]
            GNB.fit(X_prain_fold, y_train_fold)
            predict = GNB.predict(X_pest_fold)
            print("NB Acc: ",metrics.balanced_accuracy_score(y_test_fold, predict))

    ax[0,0].set_title("1")
    ax[0,0].set_ylabel("1")
    plt.tight_layout()
    fig_file_name = filename.split(".")[0]+"_PCA_LDA_original.png"
    plt.savefig(fig_file_name)