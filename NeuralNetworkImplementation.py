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

model = Sequential(
    [
        tf.keras.Input(shape = (8,)),
        Dense(units=600, activation="relu"),
        Dense(units=575, activation="relu"),
        Dense(units=550, activation="relu"),
        Dense(units=525, activation="relu"),
        Dense(units=500, activation="relu"),
        Dense(units=475, activation="relu"),
        Dense(units=450, activation="relu"),
        Dense(units=425, activation="relu"),
        Dense(units=400, activation="relu"),
        Dense(units=375, activation="relu"),
        Dense(units=350, activation="relu"),
        Dense(units=325, activation="relu"),
        Dense(units=300, activation="relu"),
        Dense(units=275, activation="relu"),
        Dense(units=250, activation="relu"),
        Dense(units=200, activation="relu"),
        Dense(units=175, activation="relu"),
        Dense(units=150, activation="relu"),
        Dense(units=125, activation="relu"),
        Dense(units=100, activation="relu"),
        Dense(units=75, activation="relu"),
        Dense(units=65, activation="relu"),
        Dense(units=50,activation="relu"),
        Dense(units=38,activation="relu"),
        Dense(units=25,activation="relu"),
        Dense(units=15,activation="relu"),
        Dense(units=1,activation="sigmoid"),
    ],name = "mymodel"
)

print(model.summary())

'''[layer1,layer2,layer3] = model.layers
W1,b1 = layer1.get_weights();
W2,b2 = layer2.get_weights();
W3,b3 = layer3.get_weights();'''

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.001),
)

model.fit(x_train_norm,y_train,epochs = 650)

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

