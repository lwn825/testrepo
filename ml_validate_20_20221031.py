import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('/storage_1/S4106037004/ShiPing/data_csv_weight_20220301/parity3/split_33_67/data_weight_20220302.csv')
data = data.drop([95,149])
data = data.reset_index(drop = True)
data = data.drop(["ADG_round"],axis = 1)
X = data.iloc[:,1:38]
y = data["sow_parity3_Hliveborn"]
sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 5566)

# set the machine learning alogrithm
lda = LinearDiscriminantAnalysis()
svm = SVC()
ada = AdaBoostClassifier(n_estimators = 5000, learning_rate=0.001, random_state = 5566)
gb = GradientBoostingClassifier(n_estimators = 5000, learning_rate=0.001, random_state = 5566)
lr = LogisticRegression(C = 0.01, max_iter=5000, random_state = 5566)
nn = MLPClassifier(max_iter = 5000)
rf = RandomForestClassifier(n_estimators = 5000,oob_score=True, warm_start = False, n_jobs = -1, random_state = 5566)
estimators = [('lda', lda), ('svm', svm), ('ada',ada), ('gb',gb), ('lr', lr), ('nn', nn)]
sc = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
ml = [lda, svm, ada, gb, lr, nn, rf, sc]
ml_name = ["lda", "svm", "ada", "gb", "lr", "nn", "rf", "sc"]

# strat cross validation
ml_l = []
pred_l = []
true_l = []
acc_l = []
sen_l = []
spe_l = []
pre_l = []
acc_l = []
for i in range(len(ml)):
    for train_index, test_index in sss.split(X, y):
        train_x, train_y = X.iloc[train_index], y.iloc[train_index]
        test_x, test_y = X.iloc[test_index], y.iloc[test_index]
        model = ml[i]
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        pred_l.append(pred)
        true_l.append(test_y)
        acc_l.append(model.score(test_x, test_y))
        sen_l.append(recall_score(test_y, pred, pos_label=1))
        spe_l.append(recall_score(test_y, pred, pos_label=0))
        pre_l.append(precision_score(test_y, pred, pos_label=1))
        ml_l.append(model)
result_dic = {"model": ml_l, "accuracy": acc_l, "precision": pre_l, "sensitivity": sen_l, "specificity": spe_l, "true_y": true_l, "predict": pred_l}
result_df = pd.DataFrame(result_dic)
result_df.to_csv("/storage_1/S4106037004/ShiPing/result/result_crossValidate_20220822/validation_Hlive_20221031.csv", index = False)
result_df.to_pickle("/storage_1/S4106037004/ShiPing/result/result_crossValidate_20220822/validation_Hlive_20221031.pickle")
