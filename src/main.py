import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential


class Constants:
    time_step = 0.01
    inp_size = (8,1,1)
    max_time = 15
    velo_v_max = 60
    thes_v_max = 50


class Main:
    
    def runRound(self, predator, prey, trials) :
        lil = None
        

    def runMain(self, loadFile, saveFile, rounds) :

        constants = Constants()
       # print(tf._kernel_dir.)
        
        if (loadFile == None or len(loadFile) == 0) :
            # create new model
            predator = Sequential([
                Input(shape=constants.inp_size),
                Dense(units=84, activation="relu"),
                Dense(units=10, activation="softmax"),
            ])
            
            prey = Sequential([
                Input(shape=constants.inp_size),
                Dense(units=84, activation="relu"),
                Dense(units=10, activation="softmax"),
            ])

        else :
            # load old model
            model = None
            
        self.runRound(predator, prey, rounds)
        
        
main = Main()
main.runMain(1,2,1)