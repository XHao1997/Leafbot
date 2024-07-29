
from module.robot import Robot
import numpy as np
import os
import time
import copy
# Get the directory of the main Python script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_directory)
robot = Robot()
robot.initialize_robot()
start_pose = {'RPY':[-np.pi/2,0,0],
              'XYZ':[0.12,0.18,0.067+0.04145]}
# end_pose = {'RPY':[-np.pi/3,0,0],
#             'XYZ':[0,0.15,0.15]}
# q_intermediate = robot.plan_straight_trajectory(start_pose,end_pose)
# robot.move_along_trajectory(q_intermediate)
# robot.move_to(start_pose)
robot.move_servo_to([90, 0, 90, 90, 90])
print(robot.get_joints_list())
# robot.move_gripper(100)
# time.sleep(5)
# robot.move_gripper(140)

# start_joint_position = np.array([90, 0, 90, 90, 90])
# end_joint_position = np.array([30, 0, 90, 90, 90])
# intermediate_joint1_position = np.linspace(90,30,8)
# for joint1 in intermediate_joint1_position:
#     robot.move_servo_to(np.array([joint1, 0, 90, 90, 90]))
#     time.sleep(1)
