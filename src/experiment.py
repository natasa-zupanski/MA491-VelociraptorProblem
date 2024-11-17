import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.losses import Loss
import numpy as np

#from tensorflow.python.keras.losses import Reduction

model = Sequential([
                InputLayer(input_shape=(2,)),
                Dense(units=84, input_dim=1, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=1, activation="relu"),
            ])

X  = [[2,1],[1,2]]
X = np.array(X)
print(X.shape)


class CustomLoss(Loss) :
    _name_scope = "custom_loss"
    reduction='auto'
    
    def __init__(self, num) :
        self.num = num
    
    def call(self, y_true, y_pred):
        # have y_true be the change in elo?
        # have y_pred be the steps taken
        diff = float(y_true) - float(y_pred)
        print(diff.shape)
        print(y_true)
        print(y_pred)
        return self.num
    
    
model.compile(
    optimizer='rmsprop')

print("FUCK")

model.train_on_batch(X, 1*X)

print("HELL")