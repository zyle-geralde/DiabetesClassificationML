import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix,classification_report
import pandas as pd


def load_data():
    data = pd.read_csv("diabetes2.csv")

    array = data.values
    train_arr = array[:537,:]
    test_arr = array[537:,:]
    x_train = train_arr[:,:8]
    y_train = train_arr[:,8]
    x_test = test_arr[:,:8]
    y_test = test_arr[:,8]

    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = load_data()

std = StandardScaler()
X_norm = std.fit_transform(x_train);

lr_model = LogisticRegression()
lr_model.fit(X_norm,y_train)

print(lr_model.intercept_)
print(lr_model.coef_)
print(lr_model.get_params()['C'])
print(lr_model.n_iter_)

xnorm_test = std.transform(x_test)
y_pred = lr_model.predict(xnorm_test)



'''Accuracy'''
print(accuracy_score(y_test,y_pred))

'''Calculate ROC AND Plot ROC'''
yy = lr_model.predict_proba(xnorm_test)
y_prob = yy[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

'''Confusion matrix'''
print(confusion_matrix(y_test,y_pred))
print(pd.crosstab(y_test,y_pred,rownames=["Actual"],colnames=["Predicted"]))

'''Classification Report'''

print(classification_report(y_test,y_pred))

