import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer, Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model

TF_ENABLE_ONEDNN_OPTS=0

class Constants:
    time_step = 0.01
    inp_size = (1, 11)
    max_time = 15
    velo_v_max = 60*1000/3600
    thes_v_max = 50*1000/3600
    velo_tr_min = 1.5
    thes_tr_min = 0.5
    reach = 0.8
    a_max = 2
    turn_max = 10000
    precision = 10
    
constants = Constants()

class Dinosaur:
    velocity = 0
    acceleration = 0
    direction = np.pi/2
    turn_radius = 0

    def __init__(self, loc, tr_min, v_max):
        self.positions = [loc]
        self.tr_min = tr_min
        self.v_max = v_max
        self.a_max = v_max**2/tr_min
    
    def getInfo(self):
        return [self.positions[-1][0], self.positions[-1][1], self.velocity, self.acceleration, self.direction]
    
    def advance(self, acceleration, turn_radius) :
        self.acceleration = acceleration*constants.a_max
        self.turn_radius = turn_radius*constants.turn_max
        self.velocity = max(min(self.velocity + self.acceleration*constants.time_step,self.v_max),0)
        if self.velocity == 0: # turn around
            self.direction = np.mod(self.direction + np.pi,2*np.pi)
            return

        turn_dir = np.sign(turn_radius)
        self.turn_radius = max(self.velocity**2/self.a_max,abs(self.turn_radius))
        if abs(self.turn_radius)/constants.turn_max <= .75:
            dtheta = self.velocity*constants.time_step/self.turn_radius
            loc = [np.round(self.positions[-1][0] + self.turn_radius*(np.sin(self.direction)*turn_dir+np.cos(self.direction-(dtheta-np.pi/2)*turn_dir)),constants.precision),
                    np.round(self.positions[-1][1] + self.turn_radius*(-1*np.cos(self.direction)*turn_dir+np.sin(self.direction-(dtheta-np.pi/2)*turn_dir)),constants.precision)]
            self.positions.append(loc)
            self.direction = np.mod(self.direction + dtheta,2*np.pi)
        else:
            loc = [np.round(self.positions[-1][0] + self.velocity*constants.time_step*np.cos(self.direction),constants.precision),
                   np.round(self.positions[-1][1] + self.velocity*constants.time_step*np.sin(self.direction),constants.precision)]
    
    # def model(self, X, hsize=[84, 60, 16]) :
    #     hs = []
    #     for i in range(hsize.count) :
    #         if (i == 0) :
    #             h = Dense(X, hsize[i], activation="relu")
    #             hs += h
    #         else:
    #             h = Dense(hs[i-1], hsize[i], activation="relu")
    #             hs += h
    #     return hs[len(hs)-1], hs[len(hs)-2]        

class Model:
    time = 0

    def __init__(self):
        self.velo = Dinosaur([0,0],constants.velo_tr_min,constants.velo_v_max)
        self.thes = Dinosaur([0,15],constants.thes_tr_min,constants.thes_v_max)

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
    trial_count = 0
    
    def runRound(self, predator, prey, trials) :
        for i in range(trials):
            if i < 100 :
                random = True
            else :
                random = False
            self.runTrial(predator, prey, random)
    
    def runTrial(self, predator, prey, random):
        predator.compile(optimizer='adam',
              loss='mse')
        prey.compile(optimizer='adam',
              loss='mse')
        
        m = Model()
        past_info = []
        past_prey_preds = []
        past_pred_preds = []
        #step = 0
        while m.endConditionsMet() == 0:
            info = [m.getInfo()]
            #print(info)
            #info.reshape(constants.inp_size)
            info = tf.convert_to_tensor(info)
            past_info.append(info)
            
            pred_predict = predator(info)
            # if (random) :
            #     rands = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            #     predict = pred_predict[0] + rands
            #     predict[0] = min(1, max(-1, predict[0]))
            #     predict[1] = min(1, max(-1, predict[1]))
            #     pred_predict = [predict]
            #print(pred_predict)
            past_pred_preds.append(pred_predict)
            prey_predict = prey(info)
            past_prey_preds.append(prey_predict)
            pred_predict = np.array(pred_predict).flatten()
            prey_predict = np.array(prey_predict).flatten()
            m.advanceModel(pred_predict, prey_predict)
            # step += 1
            # if step % 100 :
            #     temp_info = np.array(past_info)
            #     prey.train_on_batch(tf.convert_to_tensor(temp_info), tf.convert_to_tensor(self.getIdeal3(temp_info)))
            #     predator.train_on_batch(tf.convert_to_tensor(temp_info), tf.convert_to_tensor(self.getIdeal2(temp_info)))

        past_info = np.array(past_info)
        if m.endConditionsMet() == 1 :
            # prey won
            prey.train_on_batch(tf.convert_to_tensor(past_info), tf.convert_to_tensor(past_prey_preds))
            predator.train_on_batch(tf.convert_to_tensor(past_info), tf.convert_to_tensor(self.getIdeal2(past_info)))
        elif m.endConditionsMet() == 2 :
            # predator won
            self.velo_wins += 1
            predator.train_on_batch(past_info, tf.convert_to_tensor(past_pred_preds))
            prey.train_on_batch(past_info, tf.convert_to_tensor(self.getIdeal3(past_info)))
        self.trial_count += 1
        if (self.trial_count % 50 == 0) :
            self.display_paths(m.velo, m.thes)
            #predator.save_weights("./pred_checks/my_checkpoint", overwrite=True)
            #prey.save_weights("./prey_checks/my_checkpoint", overwrite=True)

        
        #past_info=np.array(past_info)
        #predator.train_on_batch(past_info, pred_mult*np.array(past_pred_preds))
        #prey.train_on_batch(past_info, prey_mult*np.array(past_prey_preds))
        
    def getIdeal(self, len) :
        res = []
        for i in range(len) :
            res.append([[constants.a_max,constants.turn_max]])
        return res
    
    def getIdeal2(self, past_info) :
        #print(len(past_info))
        res = []
        if (len(past_info) == 0) :
            print("wtf")
            return res
        if (len(past_info) == 1) :
            res.append([[constants.a_max,constants.turn_max]])
            return res
        for i in range(len(past_info)-1) :
            last = past_info[i][0]
            next = past_info[i+1][0]
            last_loc = [last[0], last[1]]
            last_dir = last[4]
            next_loc = [next[5], next[6]]
            v = (next_loc[0]-last_loc[0])/np.cos(last_dir)
            yhat = last_loc[1] + v * np.sin(last_dir)
            if abs(yhat - next_loc[1]) < (constants.reach * 3) :
                res.append([[constants.a_max, constants.turn_max]])
            else :
                res.append([[constants.a_max, 0]])
        res.append([[constants.a_max, 0]])
        return res
    
    def getIdeal3(self, past_info) :
        res = []
        if (len(past_info) == 0) :
            return res
        if (len(past_info) == 1) :
            res.append([[constants.a_max,constants.turn_max]])
            return res
        for i in range(len(past_info)-1) :
            last = past_info[i][0]
            next = past_info[i+1][0]
            last_loc = [last[0], last[1]]
            next_loc = [next[5], next[6]]
            distance = np.sqrt(np.square(last_loc[0]-next_loc[0]) + np.square(last_loc[1]-next_loc[1]))
            if distance < constants.reach * 2 :
                res.append([[constants.a_max, 0]])
            else :
                res.append([[constants.a_max, constants.turn_max]])
        res.append([[constants.a_max, constants.turn_max]])
        return res    

    def runMain(self, loadFile, saveFile, trials) :
       # print(tf._kernel_dir.)
        
        # create new model
        predator = Sequential([
            InputLayer(input_shape=constants.inp_size),
            #Dense(units=1, input_shape=constants.inp_size),
            Dense(units=84, input_shape=constants.inp_size),
            Dense(units=60),
            Dense(units=2),
        ])
        predator.summary()
            
        prey = Sequential([
            InputLayer(input_shape=constants.inp_size),
            #Dense(units=1, input_shape=constants.inp_size),
            Dense(units=84, input_shape=constants.inp_size),
            Dense(units=60),
            Dense(units=2),
        ])

        if (loadFile != None) :
            # load old model
            print("Loading")
            predator.compile(optimizer='adam',
              loss='mse')
            prey.compile(optimizer='adam',
              loss='mse')
            predator.train_on_batch(tf.convert_to_tensor([Model().getInfo()]), tf.convert_to_tensor([[constants.a_max, 0]]))
            prey.train_on_batch(tf.convert_to_tensor([Model().getInfo()]), tf.convert_to_tensor([[constants.a_max, 0]]))
            predator = load_model("./pred_checks/my_pred.keras")
            prey = load_model("./prey_checks/my_prey.keras")
            
        self.runRound(predator, prey, trials)
        print("Velo wins: " + str(self.velo_wins))
        print("Thes wins: " + str(trials - self.velo_wins))
        predator.save("./pred_checks/my_pred.keras", save_format='tf')
        prey.save("./prey_checks/my_prey.keras", save_format='tf')

    def display_paths(self, predator, prey):
        pred_pos = np.array(predator.positions)
        prey_pos = np.array(prey.positions)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.plot(pred_pos[:,0],pred_pos[:,1], '.b-', label='predator')
        ax1.plot(prey_pos[:,0],prey_pos[:,1], '.r-', label='prey')
        plt.legend(loc='upper left')
        plt.show()
        
main = Main()
main.runMain(None,None,100)