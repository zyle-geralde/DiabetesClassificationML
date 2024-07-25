import numpy as np
import matplotlib as pyplot
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report
np.set_printoptions(edgeitems=2000, linewidth=2000)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def load_data():
    data = pd.read_csv("diabetes2.csv")
    array = data.values
    train_arr = array[:537,:]
    test_arr = array[537:,:]
    x_train = train_arr[:,:8]
    y_train = train_arr[:,8]
    x_test = test_arr[:,:8]
    y_test = test_arr[:,8]
    allX = array[:,:8]
    allY = array[:,8]
    return allX,allY,x_train,y_train,x_test,y_test

allX,allY,x_train,y_train,x_test,y_test = load_data()

norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(x_train)
x_train_norm = norm_l(x_train)

from tensorflow.keras.layers import Dropout

model = Sequential([
    tf.keras.Input(shape=(8,)),
    Dense(units=128, activation="sigmoid"),
    Dense(units=62, activation="sigmoid"),
    Dense(units=1, activation="sigmoid"),
], name="mymodel")


print(model.summary())

'''[layer1,layer2,layer3] = model.layers
W1,b1 = layer1.get_weights();
W2,b2 = layer2.get_weights();
W3,b3 = layer3.get_weights();'''

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.001),
)

model.fit(x_train_norm,y_train,epochs = 100)

x_test_norm = norm_l(x_test)
prediction = model.predict(x_test_norm)
y_pred = (prediction>=0.5).astype(int)

'''Accuracy Score'''
print(accuracy_score(y_test.flatten(),y_pred.flatten()))#.flatten() to ensure that it is a 1-d array

print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

'''Confusion matrix'''
print(confusion_matrix(y_test.flatten(),y_pred.flatten()))
print(pd.crosstab(y_test.flatten(),y_pred.flatten(),rownames=["Actual"],colnames=["Predicted"]))

'''Classification Report'''

print(classification_report(y_test.flatten(),y_pred.flatten()))

