import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras import Sequential

TF_ENABLE_ONEDNN_OPTS=0

def distance(pos1, pos2) :
    return np.sqrt(np.square(pos1[0]-pos2[0]) + np.square(pos1[1]-pos2[1]))

def getMaxIndex(np_arr) :
    value = max(np_arr)
    indices = np.where(np_arr == value)
    length = len(indices)
    if length == 1 :
        return indices[0]
    else :
        return np.random.randint(length)
    
def getMaxVersion(np_arr) :
    index = getMaxIndex(np_arr)
    arr_base = np.zeros(len(np_arr))
    arr_base[index] = 1
    return arr_base

def getMaxOverResults(np_arr) :
    if np_arr.ndim != 1 :
        shape = np_arr.shape
        arr_base = np.zeros(shape)
        for i in range(shape[0]) :
            arr_base[i] = getMaxOverResults(np_arr[i])
        return arr_base
    else :
        return getMaxVersion(np_arr)
    
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
        self.a_tr_max = v_max**2/tr_min
    
    def getInfo(self):
        return [self.positions[-1][0], self.positions[-1][1], self.velocity, self.acceleration, self.direction]
    
    def advance(self, choice) :
        if choice == 0 :
            # speed up
            self.acceleration = constants.a_max
            turn_radius = None
        elif choice == 1 :
            # slow down
            self.acceleration = -1 * constants.a_max
            turn_radius = None
        else :
            turn_radius = (np.square(self.velocity)) / self.a_tr_max
            # self.turn_radius = max(self.velocity**2/self.a_max,abs(self.turn_radius)) # shouldn't be needed but keep if buggy
            self.acceleration = 0
            turn_dir = 1
            if choice == 2 :
                # turn left
                turn_radius *= -1
                turn_dir = -1
        
        distance = self.velocity * constants.time_step
        if turn_radius == None :
            # go straight
            self.positions.append([self.positions[-1][0] + distance * np.cos(self.direction), self.positions[-1][1] + distance * np.sin(self.direction)])
            self.velocity += self.acceleration * constants.time_step
            self.velocity = max(self.velocity, self.v_max)
        else :
            # turn
            if (turn_radius != 0) :
                dtheta = distance/turn_radius
            else :
                dtheta = np.pi
            self.positions.append([self.positions[-1][0] + self.turn_radius*(np.sin(self.direction)*turn_dir+np.cos(self.direction-(dtheta-np.pi/2)*turn_dir)),
                self.positions[-1][1] + self.turn_radius*(-1*np.cos(self.direction)*turn_dir+np.sin(self.direction-(dtheta-np.pi/2)*turn_dir))])
            self.direction += dtheta
            self.direction = np.mod(self.direction, 2*np.pi)  

class Model:
    time = 0

    def __init__(self):
        self.velo = Dinosaur([0,0],constants.velo_tr_min,constants.velo_v_max)
        self.thes = Dinosaur([0,15],constants.thes_tr_min,constants.thes_v_max)

    def endConditionsMet(self):
        if distance(self.velo.positions[-1], self.thes.positions[-1]) <= constants.reach :
            return 2
        elif self.time >= constants.max_time:
            return 1
        return 0
    
    def getInfo(self):
        return self.velo.getInfo() + self.thes.getInfo() + [self.time]
    
    def advanceModel(self, velo_decision, thes_decision):
        self.velo.advance(getMaxIndex(velo_decision))
        self.thes.advance(getMaxIndex(thes_decision))
        self.time += constants.time_step # increment time

class Main:
    velo_wins = 0
    trial_count = 0
    
    def runRound(self, predator, prey, trials) :
        for i in range(trials):
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
        #step = 0
        while m.endConditionsMet() == 0:
            info = tf.convert_to_tensor(np.array([m.getInfo()]).astype('float32'))
            past_info.append(info)
            
            pred_predict = predator(info)
            prey_predict = prey(info)
            
            past_pred_preds.append(pred_predict)
            past_prey_preds.append(prey_predict)
            
            pred_predict = np.array(pred_predict).flatten()
            prey_predict = np.array(prey_predict).flatten()
            
            m.advanceModel(pred_predict, prey_predict)

        arr_info = np.array(past_info)
        past_info = tf.convert_to_tensor(past_info)
        if m.endConditionsMet() == 1 :
            # prey won
            past_prey_preds = getMaxOverResults(np.array(past_prey_preds))
            past_prey_preds = tf.convert_to_tensor(past_prey_preds)
            prey.train_on_batch(past_info, past_prey_preds)
            predator.train_on_batch(past_info, tf.convert_to_tensor(self.getIdeal(arr_info)))
        elif m.endConditionsMet() == 2 :
            # predator won
            self.velo_wins += 1
            past_pred_preds = getMaxOverResults(np.array(past_pred_preds))
            past_pred_preds = tf.convert_to_tensor(past_pred_preds)
            predator.train_on_batch(past_info, past_pred_preds)
            prey.train_on_batch(past_info, tf.convert_to_tensor(self.getIdeal(arr_info)))
    
        self.trial_count += 1
        if (self.trial_count % 10 == 0) :
            self.display_paths(m.velo, m.thes)
        
    def getIdeal(self, arr) :
        res = []
        for i in range(len(arr)) :
            res.append([[1,0,0,0]])
        return res  

    def runMain(self, loadFile, saveFile, trials) :
        
        # create new model
        predator = Sequential([
            InputLayer(input_shape=constants.inp_size),
            Dense(units=84, input_shape=constants.inp_size, activation='relu'),
            Dense(units=60, activation='relu'),
            Dense(units=4, activation='softmax')
        ])
        predator.summary()
            
        prey = Sequential([
            InputLayer(input_shape=constants.inp_size),
            Dense(units=84, input_shape=constants.inp_size, activation='relu'),
            Dense(units=60, activation='relu'),
            Dense(units=4, activation='softmax')
        ])

        if (loadFile != None) :
            # load old model
            print("Loading")
            predator.compile(optimizer='adam',
             loss='categorical_crossentropy')
            prey.compile(optimizer='adam',
             loss='categorical_crossentropy')
            predator.train_on_batch(tf.convert_to_tensor([Model().getInfo()]), tf.convert_to_tensor([[1,0,0,0]]))
            prey.train_on_batch(tf.convert_to_tensor([Model().getInfo()]), tf.convert_to_tensor([[1,0,0,0]]))
            predator.load_weights("./pred_checks/my_pred2")
            prey.load_weights("./prey_checks/my_prey2")
            
        self.runRound(predator, prey, trials)
        print("Velo wins: " + str(self.velo_wins))
        print("Thes wins: " + str(trials - self.velo_wins))
        predator.save_weights("./pred_checks/my_pred2")
        prey.save_weights("./prey_checks/my_prey2")

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