import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras import Sequential


class Constants:
    time_step = 0.01
    inp_size = (1,1,11)
    max_time = 15
    velo_v_max = 60
    thes_v_max = 50
    velo_tr_min = 1.5
    thes_tr_min = 0.5
    
constants = Constants()

class Dinosaur:
    speed = 0
    acceleration = 0
    direction = 0
    turn_radius = 0

    def __init__(self, loc, tr_min, v_max):
        self.location = loc
        self.tr_min = tr_min
        self.v_max = v_max
    
    def getInfo(self):
        return [self.location[0], self.location[1], self.speed, self.acceleration, self.direction]
    
    def advance(self, acceleration, turn_radius):
        self.acceleration = acceleration
        self.turn_radius = max(turn_radius,self.tr_min)

        self.speed = np.sqrt(self.turn_radius*self.acceleration)
        
        self.speed += self.acceleration*constants.time_step
        self.location[0] += self.speed*constants.time_step*np.cos(self.direction)
        self.location[1] += self.speed*constants.time_step*np.sin(self.direction)
        
    def model(self, X, hsize=[84, 60, 16]) :
        hs = []
        for i in range(hsize.count) :
            if (i == 0) :
                h = Dense(X, hsize[i], actovation="relu")
                hs += h
            else :
                h = Dense(hs[i-1], hsize[i], activation="relu")
                hs += h
        return hs[len(hs)-1], hs[len(hs)-2]
        

class Model:
    time = 0

    def __init__(self, constants):
        self.velo = Dinosaur([0,0],constants.velo_tr_min,constants.velo_v_max)
        self.thes = Dinosaur([0,15],constants.thes_tr_min,constants.velo_v_max)
        self.constants = constants

    def endConditionsMet(self):
        print("H")
        if self.velo.location == self.thes.location or self.time >= self.constants.max_time:
            print(self.time)
            print("NO")
            return self.time
        print("YES")
        return 0
    
    def getInfo(self):
        return self.velo.getInfo() + self.thes.getInfo() + [self.time]
    
    def advanceModel(self, velo_decision, thes_decision):
        self.velo.advance(velo_decision[0], velo_decision[1])
        self.thes.advance(thes_decision[0], thes_decision[1])

class Main:
    def runRound(self, predator, prey, trials) :
        print("H")
        for _ in range(trials):
            print("H")
            self.runTrial(predator, prey)
    
    def runTrial(self, predator, prey):
        print("H")
        m = Model(constants)
        past_info = []
        past_prey_preds = []
        past_pred_preds = []
        prey_mult = 0
        pred_mult = 0
        print("H")
        while not m.endConditionsMet() > 0:
            print("F")
            info = m.getInfo()
            past_info += info
            info = [[info]]
            info = np.array(info)
            info.reshape(constants.inp_size)
            #print(np.array(info).shape)
            info = tf.convert_to_tensor(np.array(info))
            print(info)
            print(info.shape)
            
            #pred_predict = predator.predict(info)
            #prey_predict = prey.predict(info)
            pred_predict = predator(info)
            past_pred_preds.append(pred_predict)
            prey_predict = prey(info)
            past_prey_preds.append(prey_predict)
            print("M")
            print(pred_predict)
            pred_predict = np.array(pred_predict).flatten()
            prey_predict = np.array(prey_predict).flatten()
            m.advanceModel(pred_predict, prey_predict)
        if m.endConditionsMet() == 1 :
            # prey won
            prey_mult = 1
            pred_mult = -0.5
        elif m.endConditionsMet() == 2 :
            # predator won
            pred_mult = 1
            prey_mult = -0.5
        predator.train_on_batch(past_info, pred_mult*past_pred_preds)
        prey.train_on_batch(past_info, prey_mult*past_prey_preds)

    def runMain(self, loadFile, saveFile, trials) :

        constants = Constants()
       # print(tf._kernel_dir.)
        
        if (loadFile == None or len(loadFile) == 0) :
            # create new model
            predator = Sequential([
                InputLayer(input_shape=constants.inp_size),
                Dense(units=84, input_dim=1, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=2, activation="relu"),
            ])
            
            prey = Sequential([
                InputLayer(input_shape=constants.inp_size),
                Dense(units=84, input_dim=1, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=2, activation="relu"),
            ])

        else :
            # load old model
            model = None
            
        self.runRound(predator, prey, trials)
        
        
main = Main()
main.runMain(None,None,1)