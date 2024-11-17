import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer, Flatten
from tensorflow.python.keras import Sequential

class Constants:
    time_step = 0.01
    inp_size = (1, 11)
    max_time = 15
    velo_v_max = 60*1000/3600
    thes_v_max = 50*1000/3600
    velo_tr_min = 1.5
    thes_tr_min = 0.5
    reach = 0.8
    
constants = Constants()

class Dinosaur:
    velocity = 0
    acceleration = 0
    direction = 0
    turn_radius = 0

    def __init__(self, loc, tr_min, v_max):
        self.positions = [loc]
        self.tr_min = tr_min
        self.v_max = v_max
        self.a_max = v_max**2/tr_min
    
    def getInfo(self):
        return [self.positions[-1][0], self.positions[-1][1], self.velocity, self.acceleration, self.direction]
    
    def advance(self, acceleration, turn_radius):
        self.acceleration = acceleration*self.a_max # acceleration is in [-1,1] range
        self.turn_radius = turn_radius
        self.velocity = max(min(self.velocity + self.acceleration*constants.time_step,self.v_max),0)
        if self.velocity == 0: # turn around
            self.direction += np.pi
            return

        turn_dir = np.sign(turn_radius)
        turn_radius = max(self.velocity**2/self.a_max,abs(turn_radius))
        dtheta = self.velocity*constants.time_step/turn_radius
        loc = [self.positions[-1][0] + turn_radius*(np.sin(self.direction)*turn_dir+np.cos(self.direction-(dtheta-np.pi/2)*turn_dir)),
               self.positions[-1][1] + turn_radius*(-1*np.cos(self.direction)*turn_dir+np.sin(self.direction-(dtheta-np.pi/2)*turn_dir))]
        self.positions.append(loc)
        self.direction += dtheta
        
    def model(self, X, hsize=[84, 60, 16]) :
        hs = []
        for i in range(hsize.count) :
            if (i == 0) :
                h = Dense(X, hsize[i], actovation="relu")
                hs += h
            else:
                h = Dense(hs[i-1], hsize[i], activation="relu")
                hs += h
        return hs[len(hs)-1], hs[len(hs)-2]
        

class Model:
    time = 0

    def __init__(self):
        self.velo = Dinosaur([0,0],constants.velo_tr_min,constants.velo_v_max)
        self.thes = Dinosaur([0,15],constants.thes_tr_min,constants.velo_v_max)

    def endConditionsMet(self):
        if self.distance(self.velo.positions[-1], self.thes.positions[-1]) <= constants.reach : #self.velo.positions[-1] == self.thes.positions[-1] :
            return 2
        elif self.time >= constants.max_time:
            return 1
        return 0
    
    def distance(self, pos1, pos2) :
        return np.sqrt(np.square(pos1[0]-pos2[0]) + np.square(pos1[1]-pos2[1]))
    
    def getInfo(self):
        return self.velo.getInfo() + self.thes.getInfo() + [self.time]
    
    def advanceModel(self, velo_decision, thes_decision):
        self.velo.advance(velo_decision[0], velo_decision[1])
        self.thes.advance(thes_decision[0], thes_decision[1])
        self.time += constants.time_step # increment time

class Main:
    velo_wins = 0
    
    def runRound(self, predator, prey, trials) :
        for _ in range(trials):
            self.runTrial(predator, prey)
    
    def runTrial(self, predator, prey):
        predator.compile(optimizer='adam',
              loss='categorical_crossentropy')
        prey.compile(optimizer='adam',
              loss='categorical_crossentropy')
        
        m = Model()
        past_info = []
        past_prey_preds = []
        past_pred_preds = []
        prey_mult = 0
        pred_mult = 0
        while m.endConditionsMet() == 0:
            info = np.array([m.getInfo()])
            print(info)
            info.reshape(constants.inp_size)
            info = tf.convert_to_tensor(np.array(info))
            past_info.append(info)
            
            pred_predict = predator(info)
            past_pred_preds.append(pred_predict)
            prey_predict = prey(info)
            past_prey_preds.append(prey_predict)
            pred_predict = np.array(pred_predict).flatten()
            prey_predict = np.array(prey_predict).flatten()
            m.advanceModel(pred_predict, prey_predict)
        past_info = np.array(past_info)
        if m.endConditionsMet() == 1 :
            # prey won
            prey.train_on_batch(past_info, np.array(past_prey_preds))
            predator.train_on_batch(past_info, np.array(self.getIdeal(len(past_info))))
        elif m.endConditionsMet() == 2 :
            # predator won
            self.velo_wins += 1
            predator.train_on_batch(past_info, np.array(past_pred_preds))
            prey.train_on_batch(past_info, np.array(self.getIdeal(len(past_info))))
        #past_info=np.array(past_info)
        #predator.train_on_batch(past_info, pred_mult*np.array(past_pred_preds))
        #prey.train_on_batch(past_info, prey_mult*np.array(past_prey_preds))
        
    def getIdeal(self, len) :
        res = []
        for i in range(len) :
            res.append([[1,0]])
        return res
    
    def getIdeal(self, past_info) :
        res = []
        for i in range(len(past_info)-1) :
            last = past_info[i]
            next = past_info[i+1]
            last_loc = 0
            
            

    def runMain(self, loadFile, saveFile, trials) :
       # print(tf._kernel_dir.)
        
        if (loadFile == None or len(loadFile) == 0) :
            # create new model
            predator = Sequential([
                InputLayer(input_shape=constants.inp_size),
                Dense(units=1, input_shape=constants.inp_size, activation="relu"),
                Dense(units=84, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=2, activation="tanh"),
            ])
            
            prey = Sequential([
                InputLayer(input_shape=constants.inp_size),
                Dense(units=1, input_shape=constants.inp_size, activation="relu"),
                Dense(units=84, activation="relu"),
                Dense(units=60, activation="relu"),
                Dense(units=2, activation="tanh"),
            ])

        else :
            # load old model
            model = None
            
        self.runRound(predator, prey, trials)
        print("Velo wins: " + str(self.velo_wins))
        print("Thes wins: " + str(trials - self.velo_wins))
        
    def display_paths(self, predator, prey):
        pred_pos = np.array(predator.positions)
        prey_pos = np.array(prey.positions)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.scatter(pred_pos[:,0],pred_pos[:,1], c='b', marker="o", label='predator')
        ax1.scatter(prey_pos[:,0],prey_pos[:,1], c='r', marker="o", label='prey')
        plt.legend(loc='upper left')
        plt.show()
        
main = Main()
main.runMain(None,None,20)