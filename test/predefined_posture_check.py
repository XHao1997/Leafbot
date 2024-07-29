from module.robot import Robot
import numpy as np
import os
import time
import copy
# Get the directory of the main Python script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_directory)
joints_list = np.load("joints_lists_for_cali.npy")
print(joints_list)