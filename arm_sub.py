from module.robot import Robot
from module.mission import Mission
import zmq
import threading
import numpy as np
arm = Robot()
arm.initialize_robot()
arm.move_to_zero_position()
b_time = 1
arm.Arm.Arm_Buzzer_On(b_time)
mission = Mission()

t1 = threading.Thread(task = mission.execute(arm))
t1.start()



            


