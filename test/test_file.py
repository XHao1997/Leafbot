import numpy as np
for i in range(10):
    with open('../data/imitation_car/cmd.txt', 'ab') as f:
        np.savetxt(f,np.array([1]), delimiter=',')