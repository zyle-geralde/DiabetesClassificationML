import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
np.set_printoptions(edgeitems=5, linewidth=100)

RANDOM_STATE = 55

def load_data():
    data = pd.read_csv("diabetes2.csv")
    alldata = data.values
    X = alldata[:,:8]
    Y = alldata[:,8]

    return X,Y



#Splittin data(60% training set, 20% cross validation set, 20% test set)
X,Y = load_data()
print(X)
print(X.shape)
print(Y.reshape(-1,1))
print(Y.shape)
x_train,x_,y_train,y_ = train_test_split(X,Y,train_size=0.60,random_state=1);
x_cv,x_test,y_cv,y_test = train_test_split(x_,y_,train_size=0.50,random_state=1)


#scaling the data sets
scaler = StandardScaler()

scaler.fit(x_train)

x_train_norm = scaler.transform(x_train)
x_cv_norm = scaler.transform(x_cv)
x_test_norm = scaler.transform(x_test)

#SMOTE for imbalance dataset
smote = SMOTE(random_state=42)
x_train_norm, y_train_smote = smote.fit_resample(x_train_norm, y_train)

accuracy_list_train = []
accuracy_list_val = []
subsamples_list = [0.0,0.11, 0.22, 0.35,0.53, 0.62, 0.78, 0.81,1]
max_depth_list = [2, 4, 8, 16, 32, 64, 80,None]
n_estimators_list = [10,25,50,75,100,200,300,500]

#for n_estimator = 50
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    xgb_model = XGBClassifier(n_estimators = n_estimators, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
                              early_stopping_rounds=10)
    xgb_model.fit(x_train_norm, y_train_smote, eval_set=[(x_cv_norm, y_cv)])

    print(xgb_model.best_iteration)
    accuracy_train = accuracy_score(xgb_model.predict(x_train_norm),y_train_smote)
    accuracy_val = accuracy_score(xgb_model.predict(x_cv_norm),y_cv)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

print(accuracy_list_train)
print(accuracy_list_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimator')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()


accuracy_list_train = []
accuracy_list_val = []

#for subsamples = 1
for subsample in subsamples_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    xgb_model = XGBClassifier(subsample = subsample, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
                              early_stopping_rounds=10)
    xgb_model.fit(x_train_norm, y_train_smote, eval_set=[(x_cv_norm, y_cv)])

    print(xgb_model.best_iteration)
    accuracy_train = accuracy_score(xgb_model.predict(x_train_norm),y_train_smote)
    accuracy_val = accuracy_score(xgb_model.predict(x_cv_norm),y_cv)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

print(accuracy_list_train)
print(accuracy_list_val)

plt.title('Train x Validation metrics')
plt.xlabel('subsamples')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(subsamples_list )),labels=subsamples_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()

#for max-depth










