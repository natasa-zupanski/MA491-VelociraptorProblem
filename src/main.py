import tensorflow as tf
from tensorflow.keras import Sequential

class Constants :
    time-step = 0.01


class Main:

    def runMain(loadFile, saveFile) :
        model = None
        if (loadFile == None or len(loadFile) == 0) :
            # create new model
            model = Sequential([
                Input(shape=(32,32,3,)),
                Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2,2)),
                Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2, 2)),
                Conv2D(filters=120, kernel_size=(5,5), padding="same", activation="relu"),
                Flatten(),
                Dense(units=84, activation="relu"),
                Dense(units=10, activation="softmax"),
            ])

        else :
            # load old model
            model = None
        
        
