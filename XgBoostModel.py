import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,precision_score,recall_score
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
max_depth_list = [2, 4, 8, 16, 32, 64, 80,90,100,150,200,225,280,300,None]
n_estimators_list = [10,25,50,75,100,200,300,500]
min_child_weight_list = [1,3,5,7,9,11,20,32,45,80,100,125,143,178,200,280,300,None]
lambda_list = [0,1,2,3,4,5,6,7,8,9,10,12,15,18,22,28,39,49,61,]

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

accuracy_list_train = []
accuracy_list_val = []

#for maxdepth = default value/do not set
for maxdepth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    xgb_model = XGBClassifier(max_depth = maxdepth, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
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
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()

#for min_child_weight
accuracy_list_train = []
accuracy_list_val = []

#for min_child_weight = default value/ do not set
for min_child in min_child_weight_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    xgb_model = XGBClassifier(min_child_weight = min_child, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
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
plt.xlabel('min_child_weight')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_child_weight_list )),labels=min_child_weight_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()



accuracy_list_train = []
accuracy_list_val = []

#for lambda(L2 regularization) = default
for lambd in lambda_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    xgb_model = XGBClassifier(reg_lambda=lambd, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
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
plt.xlabel('lambda value')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(lambda_list )),labels=lambda_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()

#Final Model
xgb_model = XGBClassifier(n_estimators = 50, subsample = 1,learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
                          early_stopping_rounds=10)
xgb_model.fit(x_train_norm, y_train_smote, eval_set=[(x_cv_norm, y_cv)])

accuracy_train = accuracy_score(xgb_model.predict(x_train_norm), y_train_smote)
accuracy_val = accuracy_score(xgb_model.predict(x_cv_norm), y_cv)
precission_train = precision_score(xgb_model.predict(x_train_norm), y_train_smote)
precision_val = precision_score(xgb_model.predict(x_cv_norm), y_cv)
recall_train = recall_score(xgb_model.predict(x_train_norm), y_train_smote)
recall_val = recall_score(xgb_model.predict(x_cv_norm), y_cv)

print(accuracy_train)
print(accuracy_val)
print(precission_train)
print(precision_val)
print(recall_train)
print(recall_val)













