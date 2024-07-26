import numpy as np
import matplotlib as pyplot
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential,regularizers,layers,models
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report, precision_score,recall_score
np.set_printoptions(edgeitems=2000, linewidth=2000)
from imblearn.over_sampling import SMOTE

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

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(x_train_norm, y_train)
print(y_train_resampled.shape)
print(X_train_resampled.shape)
from tensorflow.keras.layers import Dropout

tf.random.set_seed(1234)
# Define the model
model = models.Sequential([
    tf.keras.Input(shape=(8,)),
    layers.Dense(units=150, activation=None, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),  # Dropout added to reduce overfitting

    layers.Dense(units=65, activation=None, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    layers.Dense(units=35, activation=None, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    layers.Dense(units=1, activation='linear', kernel_regularizer=regularizers.l2(0.01)),
], name="mymodel")



print(model.summary())


model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(0.00001),
)

model.fit(X_train_resampled, y_train_resampled,epochs = 500)

x_test_norm = norm_l(x_test)
logit = model(x_test_norm)
pred = tf.nn.sigmoid(logit).numpy()
'''prediction = model.predict(x_test_norm)
probabilities = tf.sigmoid(prediction)'''
y_pred = (pred >= 0.5).astype(int)



'''Accuracy Score'''
print(accuracy_score(y_test.flatten(),y_pred.flatten()))#.flatten() to ensure that it is a 1-d array
print(precision_score(y_test.flatten(),y_pred.flatten()))
print(recall_score(y_test.flatten(),y_pred.flatten()))

print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

'''Confusion matrix'''
print(confusion_matrix(y_test.flatten(),y_pred.flatten()))
print(pd.crosstab(y_test.flatten(),y_pred.flatten(),rownames=["Actual"],colnames=["Predicted"]))

'''Classification Report'''

print(classification_report(y_test.flatten(),y_pred.flatten()))


