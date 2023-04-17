from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import StratifiedKFold
print("")

filenames = ["features_MAV.csv","features_SSC.csv","features_VAR.csv","features_WL.csv","features_ZC.csv"]
test_size=0.5

for filename in filenames:
    print("")
    print(filename)
    dataset = pd.read_csv("Code/results/"+filename, sep=",", decimal=".", header=None)
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values.astype(int)
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=11)
    splits = kfold.split(X,y)


    GNB = GaussianNB()

    for n,(train_index,test_index) in enumerate(splits):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        GNB.fit(x_train_fold, y_train_fold)
        predict = GNB.predict(x_test_fold)
        ###Evaluating Prediction Accuracy
        print("NB Acc: ",metrics.accuracy_score(y_test_fold, predict))
