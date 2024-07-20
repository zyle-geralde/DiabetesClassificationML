import numpy as np
import pandas as pd
np.set_printoptions(precision=2)
np.set_printoptions(edgeitems=20, linewidth=200)


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


np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_regularized_derev(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

'''def compute_gradiet_descent(x,y,alpha,w_init,b_init,compute_derev,cost_function,num_iter):
    return'''


