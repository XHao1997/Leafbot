import numpy as np
for i in range(10):
    with open('../data/cali/j1/cali_j1.txt', 'a') as f:
        np.savetxt(f, np.array([i]), newline='\n')