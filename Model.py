import numpy as np
import pandas as pd
import copy,math
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report
np.set_printoptions(precision=2)
np.set_printoptions(edgeitems=20, linewidth=200)
import matplotlib.pyplot as plt


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
def z_score_normalize(x):
    mean = np.mean(x,axis = 0)
    std = np.std(x,axis=0)

    xnorm = (x-mean)/std

    return xnorm,mean,std

xnorm,mean,std = z_score_normalize(x_train)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_regularized_cost(x,y,w,b,lambda_ = 1):
    m = x.shape[0]
    n = x.shape[1]

    cost = 0.0
    for i in range(m):
        z = np.dot(x[i],w)+b
        sig_val = sigmoid(z)
        loss = -y[i] * np.log(sig_val) - (1-y[i]) * np.log(1-sig_val)
        cost += loss

    cost = cost/m

    reg_val = 0.0
    for j in range(n):
        reg_val += (w[j] ** 2)

    reg_val = ( lambda_/(2 * m) ) * reg_val
    return cost+reg_val


def compute_regularized_derev(x,y,w,b,lambda_ = 1):
    m = x.shape[0]
    n = x.shape[1]

    dw = np.zeros((n,))
    db = 0.0

    for i in range(m):
        sig_val = sigmoid(np.dot(x[i],w) + b)
        err = sig_val - y[i]
        for j in range(n):
            dw[j] += err * x[i,j]
        db += err
    dw = dw/m
    db = db/m

    for i in range(n):
        dw[i] += ((lambda_/m)*w[i])

    return db,dw



def compute_gradiet_descent(x,y,alpha,w_init,b_init,compute_derev,cost_function,num_iter,lambda_):
    m = x.shape[0]
    n = x.shape[1]

    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iter):
        db,dw = compute_derev(x,y,w,b,lambda_)

        w = w - alpha * dw
        b = b - alpha * db


        if(i % math.ceil(num_iter/10) == 0):
            print(f" Cost Function: {cost_function(x,y,w,b,lambda_)}  w: {w}  b: {b}")
    return w,b

alpha  = 0.1
w_init = np.zeros((xnorm.shape[1]))
b_init = 0.0
lambda_ = 0.7

w_out,b_out = compute_gradiet_descent(xnorm,y_train,alpha,w_init,b_init,compute_regularized_derev,compute_regularized_cost,10000,lambda_)
print(f"final w: {w_out}\tfinale b:{b_out}")


def predict(x, w, b):
    z = np.dot(x, w) + b
    y_pred_prob = sigmoid(z)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)  # Convert probabilities to binary classes (0 or 1)
    return y_pred_class,y_pred_prob

xtest_norm,m,s = z_score_normalize(x_test);

y_pred,y_prob= predict(xtest_norm,w_out,b_out);



'''Accuracy Score'''
print("\nAccuracy")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

'''Calculate ROC AND Plot ROC'''
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

'''Confusion Metrix'''
print(confusion_matrix(y_test,y_pred))
print(pd.crosstab(y_test,y_pred,rownames=["Actual Labels"],colnames=["Predicted Labels"]))


'''Classification Report'''
print(classification_report(y_test,y_pred))


'''Adjust the polynomial expression of the model'''




