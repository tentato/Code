from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
print("")
matplotlib.style.use('ggplot')

main_folder = 'results/KrzysztofJ_all/'
filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

for filename in filenames:
    dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=None, 
        names=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16"])
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values.astype(int)

    class_pairs = []
    class_numbers = [*set(y)] # removes duplcates
    for i, a in enumerate(class_numbers):
        for j, b in enumerate(class_numbers):
            if a==b:
                break
            class_pairs.append([a,b]) #(NumClasses * (NumClasses – 1)) / 2

    ### Reject worst feature/s
    fig, ax = plt.subplots(9,9, figsize=(100,100))
    for pair2 in class_pairs:
        print(pair2)
        filtered_df = dataset[(dataset['c16'] == pair2[0]) | (dataset['c16'] == pair2[1])] 
        X_f = filtered_df.iloc[:, 0:-1].values
        y_f = filtered_df.iloc[:, -1].values.astype(int)
        k_2=1
        X_new = SelectKBest()
        new_X = X_new.fit_transform(X_f, y_f)
        print("Scores: ",X_new.scores_)
        # print("Min score: ", np.argmin(X_new.scores_))
        print("Pvalues: ",X_new.pvalues_)
        indexes_list = np.argpartition(X_new.scores_, k_2)
        worst_features_indexes = indexes_list[:k_2]
        print(worst_features_indexes)
        X_f = np.delete(X_f, worst_features_indexes,1)

        # PCA
        X_p = PCA(n_components=2).fit_transform(X_f)
        ax[pair2[0]-1, pair2[1]-1].scatter(*X_p.T, c=y_f, cmap="cool")
        # LDA
        X_l = LDA().fit_transform(X_f, y_f)
        ax[pair2[1]-1, pair2[0]-1].scatter(X_l, X_l, c=y_f, cmap="cool")

        if pair2[1]-1 == 0:
            ax[pair2[1]-1, pair2[0]-1].set_title(pair2[0])
        if pair2[1]-1 == 0:
            ax[pair2[0]-1, pair2[1]-1].set_ylabel(pair2[0])

        kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
        splits = kfold.split(X_l,y_f)

        # model = GaussianNB()
        model = SVC()

        for n,(train_index,test_index) in enumerate(splits):
            X_prain_fold, X_pest_fold = X_l[train_index], X_l[test_index]
            y_train_fold, y_test_fold = y_f[train_index], y_f[test_index]
            model.fit(X_prain_fold, y_train_fold)
            predict = model.predict(X_pest_fold)
            print("SVC Acc: ",metrics.balanced_accuracy_score(y_test_fold, predict))

    ax[0,0].set_title("1")
    ax[0,0].set_ylabel("1")
    plt.tight_layout()
    fig_file_name = "PCA_LDA/"+filename.split(".")[0]+"_PCA_LDA_K_Best_Features.png"
    plt.savefig(fig_file_name)