import multiprocessing.process
import time
import numpy as np
from module.msg import ARMTASK, Msg
from module.robot import Robot
import zmq
import cv2 
import threading
import time
import multiprocessing
import copy
import PyKDL as kdl
JOINT2IDS = {'J1':1,'J2':2,'J3':3,'J4':4,'J5':5,'J6':6}
def cmd2servo_angle(cmd):
    if len(cmd)==1:
        for key, value in cmd.items():
            angle = value
            id = int(JOINT2IDS[key])
    return id, angle


class Mission:

    def __init__(self):
        self.info = None
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://192.168.101.14:3333")
        self.nn_joints_list = []

    def execute(self,arm:Robot):
        while True:
            self.info = self.socket.recv_pyobj()

            if self.info.task == ARMTASK.MOVE_SINGLE_JOINT:
                self.move_single_joint(arm)
            
            elif self.info.task == ARMTASK.MOVE_ALL_JOINT:
                self.move_all_joints(arm)
            
            elif self.info.task == ARMTASK.MOVE_ZERO_POSITION:
                self.move_zero_position(arm)
            
            elif self.info.task==ARMTASK.READ_SERVO:
                respond_msg = self.read_servo(arm)
                self.socket.send_pyobj(respond_msg)

            elif self.info.task==ARMTASK.PICK_CLOSEST_LEAF:
                pick_index = self.pick_closest_leaf(arm)
                self.socket.send_pyobj(pick_index)

            elif self.info.task == ARMTASK.MOVE_ARM_MOVE_RANDOM:
                self.move_random(arm)
            
            elif self.info.task == ARMTASK.MOVE_ARM_EYE2HAND:
                self.move_random(arm,int(self.info.cmd))
                respond_msg = Msg(ARMTASK.DONE)
                self.socket.send_pyobj(respond_msg)
                time.sleep(1)

            elif self.info.task == ARMTASK.SET_TORQUE:
                self.set_torque(arm)
                arm.update_joints_list()
            
            elif self.info.task == ARMTASK.OFF_TORQUE:
                self.off_torque(arm)

            elif self.info.task == ARMTASK.IMITATION_MOVE:
                print('imitation task called')
                joint_list = next(arm.generate_nn_positions())
                joint_list[0] = 90
                arm.move_servo_to(joint_list,speed=3000)

            elif self.info.task == ARMTASK.CALI_J1:
                sign = 1 if self.info.cmd == 'l' else -1
                if self.info.cmd == 'p':
                    # sign = np.random.uniform(-3, 3)
                    arm.move_servo_by(6, 120,speed=200)
                else:
                    arm.move_servo_by(1, arm.get_joints_list()[0]+sign*2,speed=200)
                self.socket.send_pyobj(1)

            elif self.info.task == ARMTASK.CALI_J1_imitation:
                sign = 1 if self.info.cmd == 'l' else -1
                if self.info.cmd == 'p':
                    sign = int(np.random.uniform(4, 4))
                
                arm.move_servo_by(1, arm.get_joints_list()[0]+sign*3,speed=200)
                self.socket.send_pyobj(1)

    def move_single_joint(self, arm:Robot):
        id, angle = cmd2servo_angle(self.info.cmd)
        arm.move_servo_by(id=id, angle=angle)
    
    def move_all_joints(self,arm:Robot):
        arm.move_servo_to(self.info.cmd)
    
    def move_zero_position(self,arm:Robot):
        arm.move_to_zero_position()
    
    def read_servo(self,arm:Robot):
        arm.update_joints_list()
        servo_angles = arm.get_joints_list()
        time.sleep(1)
        return servo_angles
    
    def pick_closest_leaf(self,arm:Robot):
        xyz_list = []
        leaf_to_pick = self.info.cmd
        for location in leaf_to_pick:
            xyz_list.append(arm.coodinate_cam2robot(location.reshape(1,-1)))
        pick_index = np.argmin(
            np.linalg.norm(np.asarray(xyz_list)-arm.solve_fk_by([90, 180, 0, 0, 90])[:3,3].T,axis=1))
        xyz = xyz_list[pick_index]
        np.savetxt('xyz_list.txt', np.asarray(xyz_list))
        # xyz = xyz_list[np.argmin(
        #     np.asarray(xyz_list)[:,0],axis=1)]
        # xyz = xyz_list[np.argmin(np.asarray(xyz_list)[:,1])]
        desired_position = {'XYZ':xyz,
                    'RPY':[-np.pi/2, 0, 0]}
        arm.move_gripper(55,speed=20)
        # status = arm.move_to(desired_position,speed = 2000,offset=5)
        solution, status = arm.solve_ik_by_euler(desired_position['RPY'],desired_position['XYZ'])
        self.last_pick(arm,solution)
        # time.sleep(1)
        # arm.move_gripper(120)
        return pick_index, status

    def move_random(self, arm:Robot, num):
        print('begin')
        np.random.seed(42)

        for i in range(num):
            time.sleep(1)
            arm.move_servo_to(next(arm.generate_nn_positions()),speed=1000)
            time.sleep(1)
            arm.update_joints_list()
            self.nn_joints_list.append(copy.deepcopy(arm.solve_fk_by(arm.get_joints_list())))
            self.socket.send_pyobj(int(1))
        np.save('data/nn_train.npy',self.nn_joints_list)

    def set_torque(self, arm:Robot):
        arm.set_torque()    

    def off_torque(self,  arm:Robot):
        arm.off_torque()

    @staticmethod
    def last_pick(arm:Robot, solution):
        f = arm.solve_fk_by(arm.radius_to_servo_degree(arm.kdl_to_np(solution),offset=3),option='kdl')
        start_pose = {'XYZ':f.p,
                    'RPY':f.M.GetRPY()}
        end_pose = copy.deepcopy(start_pose)
        f_end = kdl.Frame(kdl.Rotation.RPY(0,0,0), kdl.Vector(-0.0, -0.0, -0.02))
        f_start = kdl.Frame(kdl.Rotation.RPY(0,0,0), kdl.Vector(-0.0, -0.0, 0.0125))
        end_pose['XYZ'] = (f*f_end).p
        start_pose['XYZ'] = (f*f_start).p

        q_intermediate = arm.plan_straight_trajectory(start_pose=end_pose, end_pose=start_pose)
        arm.move_along_trajectory(q_intermediate)