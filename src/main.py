import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential


class Constants:
    time_step = 0.01
    inp_size = (11,1,1)
    max_time = 15
    velo_v_max = 60
    thes_v_max = 50
    velo_tr_min = 1.5
    thes_tr_min = 0.5

class Dinosaur:
    speed = 0
    acceleration = 0
    direction = 0
    turn_radius = 0

    def __init__(self, loc, constants, tr_min, v_max):
        self.constants = constants
        self.location = loc
        self.tr_min = tr_min
        self.v_max = v_max
    
    def getInfo(self):
        return [self.location[0], self.location[1], self.speed, self.acceleration, self.direction]
    
    def advance(self, acceleration, turn_radius):
        self.acceleration = acceleration
        self.turn_radius = max(turn_radius,self.tr_min)

        self.speed = np.sqrt(self.turn_radius*self.acceleration)
        
        self.speed += self.acceleration*self.constants.time_step
        self.location[0] += self.velocity*self.constants.time_step*np.cos(self.direction)
        self.location[1] += self.velocity*self.constants.time_step*np.sin(self.direction)

class Model:
    time = 0

    def __init__(self, constants):
        self.velo = Dinosaur([0,0],constants.velo_tr_min,constants.velo_v_max)
        self.thes = Dinosaur([0,15],constants.thes_tr_min,constants.velo_v_max)
        self.constants = constants

    def endConditionsMet(self):
        if self.velo.location == self.thes.location or self.time == self.constants.max_time:
            return self.time
        return 0
    
    def getInfo(self):
        return self.velo.getInfo() + self.thes.getInfo() + [self.time]
    
    def advanceModel(self, velo_decision, thes_decision):
        self.velo.advance(velo_decision[0], velo_decision[1])
        self.thes.advance(thes_decision[0], thes_decision[1])

class Main:
    def runRound(self, predator, prey, trials) :
        for _ in range(trials):
            self.runTrial(predator, prey)
    
    def runTrial(self, predator, prey):
        m = Model()
        past_info = []
        while m.endConditionsMet() > 0:
            info = m.getInfo()
            past_info += info
            pred_predict = predator.predict(info)
            prey_predict = prey.predict(info)

            m.advanceModel(pred_predict, prey_predict)
                

    def runMain(self, loadFile, saveFile, trials) :

        constants = Constants()
       # print(tf._kernel_dir.)
        
        if (loadFile == None or len(loadFile) == 0) :
            # create new model
            predator = Sequential([
                Input(shape=constants.inp_size),
                Dense(units=84, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=2, activation="relu"),
            ])
            
            prey = Sequential([
                Input(shape=constants.inp_size),
                Dense(units=84, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=2, activation="relu"),
            ])

        else :
            # load old model
            model = None
            
        self.runRound(predator, prey, trials)
        
        
main = Main()
main.runMain(1,2,1)