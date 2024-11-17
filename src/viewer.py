import matplotlib.pyplot as plt
import numpy as np
from main import Dinosaur, constants

pred = Dinosaur([0,0],constants.velo_tr_min,constants.velo_v_max)
pred.advance(3,0)
for i in range(100):
    pred.advance(0,.55)
pred.advance(10,0)
pred.advance(-5,0)
for i in range(100):
    pred.advance(0,-.35)
pred.advance(10,0)
pred.advance(-3,0)
for i in range(100):
    pred.advance(0,1)

pos = np.array(pred.positions)

plt.plot(pos[:,0],pos[:,1], '-o')
plt.show()