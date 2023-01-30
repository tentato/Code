import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
print("")

# main_folder = 'results/KrzysztofJ_all/'
main_folder = 'results/MK/'

filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]

for filename in filenames:
    print("")
    print(filename)
    dataset = pd.read_csv(main_folder+filename, sep=",", decimal=".", header=None)
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values.astype(int)

    k=1 # remove worst features
    X_new = SelectKBest()
    new_X = X_new.fit_transform(X, y)
    # print("Scores: ",X_new.scores_)
    # print("Min score: ", np.argmin(X_new.scores_))
    # print("Pvalues: ",X_new.pvalues_)
    indexes_list = np.argpartition(X_new.scores_, k)
    worst_features_indexes = indexes_list[:k]
    print(worst_features_indexes)
    X = np.delete(X, worst_features_indexes,1)  
    print(X.shape)

    kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=11)
    splits = kfold.split(X,y)
    

    # model = SVC()
    model = GaussianNB()
    # model = KNeighborsClassifier() #best
    ovo = OneVsOneClassifier(model)

    for n,(train_index,test_index) in enumerate(splits):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        ovo.fit(x_train_fold, y_train_fold)
        predict = ovo.predict(x_test_fold)
        ###Evaluating Prediction Accuracy
        print("SVC OvO Acc: ",round(metrics.accuracy_score(y_test_fold, predict),2))