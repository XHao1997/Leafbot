#coding=utf-8
import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages/')
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
import numpy as np
from robot_utils import bezier, urdf_util
import os
from Arm_Lib import Arm_Device
import time
from scipy.spatial.transform import Rotation
from robot_utils.bezier import interpolate_orientation
import joblib
class Robot(object):
    """Represents a robot with kinematic functionality."""

    def __init__(self):
        """Initialize the Robot object."""
        self.chain = None
        self.tree = None
        self.fk_solver = None
        self.ik_solver = None
        self.trajectory = None
        self.Arm = Arm_Device()
        self.__joints_list = np.array([90, 90, 90, 90, 90, 180])
        self.predictor = {'X':None,
                          'Y':None,
                          'Z':None,
                          'J1':None}
    def __load_predictor(self):
        self.predictor['X'] = joblib.load('weights/ereg_x.pkl')
        self.predictor['Y'] = joblib.load('weights/ereg_y.pkl')
        self.predictor['Z'] = joblib.load('weights/ereg_z.pkl')
        self.predictor['J1'] = joblib.load('weights/ereg_j1.pkl')
        self.predictor['MLP'] = joblib.load('weights/mlp2.pkl')
        self.predictor['Linear'] = joblib.load('weights/lreg.pkl')

    def __read_from_urdf(self):
        """Read and parse URDF file to obtain robot model.

        Returns:
            URDF: Object representing the parsed URDF model.
        """
        project_directory = os.getcwd()
        urdf_file_path = os.path.join(project_directory, "dofbot.urdf")
        return URDF.from_xml_file(urdf_file_path)


    def __initialize_kdl_tree_from_urdf(self):
        """Initialize KDL tree from URDF model."""
        robot = self.__read_from_urdf()
        self.tree = urdf_util.kdl_tree_from_urdf_model(robot)

    def __initialize_chain(self):
        """Initialize kinematic chain from base_link to link5."""
        self.chain = self.tree.getChain("base_link", "link5")

    def __initialize_fk_solver(self):
        """Initialize forward kinematics solver for the robot."""
        chain = self.tree.getChain("base_link", "link5")
        self.fk_solver = kdl.ChainFkSolverPos_recursive(chain)

    def __initialize_ik_solver(self):
        """Initialize inverse kinematics solver for the robot."""
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain, eps=1e-6,
                                                  _maxiter = int(1e5),_eps_joints = 1e-8)
        # joint_min = kdl.JntArray(5)
        # joint_max = kdl.JntArray(5)
        # for i in range(5):
        #     joint_min[i] = -np.pi
        #     joint_max[i] = np.pi
        # fksolver = kdl.ChainFkSolverPos_recursive(self.chain)
        # iksolver_v = kdl.ChainIkSolverVel_pinv(self.chain)
        # self.ik_solver = kdl.ChainIkSolverPos_NR_JL(self.chain, joint_min, joint_max, fksolver, iksolver_v)

    def initialize_robot(self):
        """Initialize the robot by setting up KDL tree and solvers.

        This method reads the robot's information from a URDF file, initializes the KDL tree,
        sets up the kinematic chain, initializes forward kinematics solver, initializes inverse
        kinematics solver, and moves the robot to its zero position.

        """
        self.__read_from_urdf()
        self.__initialize_kdl_tree_from_urdf()
        self.__initialize_chain()
        self.__initialize_fk_solver()
        self.__initialize_ik_solver()
        self.__load_predictor()
        self.set_torque()
        time.sleep(0.1)
        self.move_to_zero_position()


    def move_to_zero_position(self):
        """
        Moves the robot arm to the zero position.
        This method updates the joints list, waits for 1 second, and then moves the servos to the zero position.
        """
        zero_position = [90, 180, 0, 0, 90, 160]

        self.move_servo_to(zero_position, 1000)
        time.sleep(1)
        self.update_joints_list()
        time.sleep(1)


    def generate_nn_positions(self):
        """Calculate end-effector positions for eye-in-hand configuration.

        Yields:
            numpy.ndarray: Current joint positions for nn model configuration.
        """
        j1 = np.random.randint(-30,30)
        j2 = np.random.randint(-60,0)
        if j2>-30:
            j3 = np.random.randint(-60,-30)
        else:
            j3 = np.random.randint(-30,0)

        j4 = -90-j2-j3
        joint_list = [90+j1,90+j2,90+j3,90+j4+8,90]
        yield joint_list        
 
    def generate_eye2hand_positions(self):
        """Calculate end-effector positions for eye-in-hand configuration.

        Yields:
            numpy.ndarray: Current joint positions for eye-in-hand configuration.
        """
        joints_lists_cali = np.load("joints_lists_for_cali.npy")
   
        for joinits_list in joints_lists_cali: 
            pass_check = self.check_limitation(joinits_list)
            if pass_check:
                yield joinits_list

    def smooth_trajectory(self, joints_list):
        """Smooth the trajectory of the robot joints.

        Args:
            joints_list (list): List of joint positions.

        Returns:
            str: Description of the method to be implemented.
        """
        # To be implemented
        return "To be implemented"

    def update_joints_list(self):
        """Update the current joints list with new values.

        Args:
            new_joints_list (list): New list of joint positions.
        """
        
        for i in range(6):
            joint = self.Arm.Arm_serial_servo_read(i+1)
            self.__joints_list[i] = joint
            time.sleep(.01)
        time.sleep(.5)

    def get_joints_list(self):
        return self.__joints_list
    
    def move_to(self, desired_position, speed=1000, option='euler', offset=0):
            """Move the robot to a specified Cartesian position.

            Args:
                desired_position (list): A list containing the desired Cartesian coordinates and orientation
                    in the format [X, Y, Z, R, P, Y], where:
                    - X, Y, Z: Cartesian coordinates specifying the position.
                    - R, P, Y: Roll, pitch, and yaw angles specifying the orientation.
                speed (int, optional): The speed of the movement in milliseconds. Defaults to 1000.
                option (str, optional): The option for solving inverse kinematics. 
                    Valid options are 'euler' and 'pose'. Defaults to 'euler'.

            Returns:
                bool: A flag indicating the success or failure of the movement.
            """
            if option == 'euler':
                solution, status = self.solve_ik_by_euler(desired_position['RPY'], desired_position['XYZ'])
            else:
                solution, status = self.solve_ik_by_pose(desired_position)
            
            self.move_servo_to(self.radius_to_servo_degree(self.kdl_to_np(solution)), speed, offset)
            time.sleep(speed/1000 + 1e-10)
            self.update_joints_list()
            return status

    def move_servo_to(self, solution, speed=1000, offset=0):
        """Rotate the servo to a specific angle."""
        pass_check = self.check_limitation(solution)
        if pass_check:
            s1 = solution[0]
            s2 = solution[1]
            s3 = solution[2]
            s4 = solution[3]+offset
            s5 = solution[4]
            if len(solution)==6:
                s6 = solution[5]
            else:
                s6 = self.get_joints_list()[-1]

            self.Arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, speed)
            time.sleep(speed/1000+1e-10)
            return 1
        else:
            print("The solution is out of robot workspace")  
            print(solution) 
            return 0
        
    def move_servo_by(self, id, angle, speed=1000):
        """
        Rotate a specific servo to a specific angle.

        Args:
            id (int): The ID of the servo to rotate.
            angle (list of float): The angle to rotate the servo to.
            speed (int, optional): The speed at which to rotate the servo. Defaults to 1000.

        Returns:
            None
        """
        self.Arm.Arm_serial_servo_write(id, angle, speed)
        time.sleep(speed/1000+1e-10)
        self.update_joints_list()

        
    def move_gripper(self, degree,speed=200):
        self.Arm.Arm_serial_servo_write(6, degree, 200)
        time.sleep(0.25)
        self.__joints_list[-1]=degree
        print(self.__joints_list[-1])
        


    def move_along_trajectory(self, q_intermediate):
        """Move the robot along a predefined trajectory."""
        # To be implemented
        q_smooth = np.zeros((5,1000))
        num_points, num_joints = q_intermediate.shape
        for i in range(5):
            q_smooth[i] = np.flip(bezier.bezier_curve(np.column_stack((np.arange(num_points),q_intermediate[:,i])),
                                                      nTimes=1000)[1])
        for i in range(1000):
            if i%10==0:  
                if i == 0:
                    speed = 1000
                    q_current = self.radius_to_servo_degree(q_smooth.T[i])
                    flag = self.move_servo_to(q_current,speed) 
                    time.sleep(5)
                else:
                    max_deg = abs(max(q_current-self.radius_to_servo_degree(q_smooth.T[i+1])))
                    speed = np.round(10*max_deg)+5
                    speed = int(speed)
                    q_current = self.radius_to_servo_degree(q_smooth.T[i])
                    flag = self.move_servo_to(q_current,speed) 
                if flag ==0:
                    break
        self.update_joints_list()
        return flag

    def plan_straight_trajectory(self,start_pose, end_pose):
        """Plan a trajectory for the robot to follow."""
        # To be implemented
        start_RPY = start_pose['RPY']
        start_XYZ = start_pose['XYZ']
        start_pose = kdl.Frame(kdl.Rotation.RPY(start_RPY[0], start_RPY[1], start_RPY[2]), 
                               kdl.Vector(start_XYZ[0],start_XYZ[1], start_XYZ[2]))
        end_RPY = end_pose['RPY']
        end_XYZ = end_pose['XYZ']
        end_pose = kdl.Frame(kdl.Rotation.RPY(end_RPY[0],end_RPY[1], end_RPY[2]), 
                             kdl.Vector(end_XYZ[0], end_XYZ[1], end_XYZ[2]))
        end_pose.M = start_pose.M
        orientation = self.kdl_to_np(end_pose.M)
        r =  Rotation.from_matrix(orientation)
        orientation = r.as_euler("xyz",degrees=False)
        # uncomment this if start_rot and end_rot is different
        # num_points = 100  # Example number of intermediate orientations
        # ori_intermediate = interpolate_orientation(start_rot, end_rot, num_points)
        start_position = start_pose.p
        end_position = end_pose.p
        # Number of intermediate points (including the endpoints)
        num_points = 10
        # Generate intermediate x, y, z values using linear interpolation
        x_intermediate = np.linspace(start_position[0], end_position[0], num_points)
        y_intermediate = np.linspace(start_position[1], end_position[1], num_points)
        z_intermediate = np.linspace(start_position[2], end_position[2], num_points)

        straightline = np.column_stack((x_intermediate,y_intermediate,z_intermediate))
        q_current = kdl.JntArray(2)
        q_intermediate = np.zeros((num_points, 5))
        for i, pos in enumerate(straightline):    
            q_current, _ = self.solve_ik_by_euler(orientation, pos)
            q_intermediate[i] = self.kdl_to_np(q_current)
        return q_intermediate
    
    def plan_circle_trajectory(self, start_pose, mid_pose, end_pose):
        start_xyz = start_pose['XYZ']
        mid_xyz = mid_pose['XYZ']
        end_xyz = end_pose['XYZ']

        start_rot = start_pose['RPY']
        end_rot = end_pose['RPY']

        # orientation = self.kdl_to_np(start_pose.M)
        # r =  Rotation.from_matrix(orientation)
        # orientation = r.as_euler("xyz",degrees=False)
        traj  = self.generate_trajectory_from_waypoints([start_xyz, mid_xyz, end_xyz])
        ori_intermediate = interpolate_orientation(start_rot, end_rot, 100)
        q_current = kdl.JntArray(5)
        q_intermediate = np.zeros((traj.shape[1], 5))
        for i, pos in enumerate(traj.T):    
            q_current, _ = self.solve_ik_by_euler(ori_intermediate[i], pos)
            q_intermediate[i] = self.kdl_to_np(q_current)
        return q_intermediate

    @staticmethod
    def generate_trajectory_from_waypoints(waypoints):
        P0 = waypoints[0]
        P1 = waypoints[1]
        P2 = waypoints[2]
        # Generate the parameter t
        t = np.linspace(0, 1, 100)

        # Calculate the points on the Bézier curve
        traj = (1-t)**2 * P0[:, np.newaxis] + 2*(1-t)*t * P1[:, np.newaxis] + t**2 * P2[:, np.newaxis]
        return traj

    def solve_ik_by_euler(self, list_RPY, list_XYZ):
        R, P, Y_ = list_RPY
        X, Y, Z = list_XYZ
        desired_pose = kdl.Frame(kdl.Rotation.RPY(R, P, Y_), kdl.Vector(X, Y, Z))
        # Solution joint positions will be stored here
        solution = kdl.JntArray(5)
        q_int = kdl.JntArray(5)
        q_int[0]  = -np.arctan2(X,Y)
        q_int[1]  = 0

        # Calculate inverse kinematics
        status = self.ik_solver.CartToJnt(q_int, desired_pose, solution)
        if solution[0]>np.pi/1.2:
            solution[0] = 0
        if abs(solution[4])>np.pi/1.5:
            solution[4] = 0 
        # if abs(solution[3])>np.pi/1.5:
        #     solution[3] = 0
        return solution, status
    
    def solve_ik_by_pose(self, desired_pose):
        # Solution joint positions will be stored here
        solution = kdl.JntArray(5)
        q_int = kdl.JntArray(5)
        print(desired_pose.p[0], desired_pose.p[1],desired_pose.p[2])
        q_int[0]  = -np.arctan2(desired_pose.p[0], desired_pose.p[1])
        # Calculate inverse kinematics
        status = self.ik_solver.CartToJnt(q_int, desired_pose, solution)
        return solution, status
    
    def solve_fk_by(self, joints_list, option='np'):
        """
        Solves the forward kinematics of the robot arm given a list of joint angles.

        Args:
            joints_list (list): A list of joint angles in degrees.

        Returns:
            numpy.ndarray: The end effector pose as a 4x4 transformation matrix.
        """
        q = kdl.JntArray(5)
        q[0] = np.deg2rad(joints_list[0]-90)
        q[1] = np.deg2rad(joints_list[1]-90)
        q[2] = np.deg2rad(joints_list[2]-90)
        q[3] = np.deg2rad(joints_list[3]-90)
        q[4] = np.deg2rad(joints_list[4]-90)
        end_effector_pose = kdl.Frame()
        self.fk_solver.JntToCart(q, end_effector_pose)
        if option == 'np':
            end_effector_pose = self.kdl_to_np(end_effector_pose)
        else:
            pass
        return end_effector_pose
    
    def generate_random_pos(self):
        """Generate a random end-effector position.
        Returns:
            kdl.Frame: Random end-effector position.
        """
        q = kdl.JntArray(5)
        q[0] = np.random.uniform(low=0, high=np.pi)
        q[1] = np.random.uniform(low=0, high=np.pi/2)
        q[2] = np.random.uniform(low=-np.pi, high=np.pi/3)
        q[3] = np.random.uniform(low=-np.pi/2, high=np.pi/2)
        q[4] = np.random.uniform(low=-np.pi/2, high=np.pi/2)
        end_effector_pose = kdl.Frame()
        self.fk_solver.JntToCart(q, end_effector_pose)
        R = np.random.uniform(low=-np.pi, high=np.pi)
        P = np.random.uniform(low=-np.pi, high=np.pi)
        Y = np.random.uniform(low=-np.pi, high=np.pi)
        random_points = end_effector_pose.p
        desired_pose = kdl.Frame(kdl.Rotation.RPY(R, P, Y), kdl.Vector(random_points[0], random_points[1], random_points[2]))
        
        return desired_pose
    
    def coodinate_cam2robot(self, cam_coordinate):
        # robot_coordinate_x = None
        # robot_coordinate_y = None
        # robot_coordinate_z = None
        # robot_coordinate_x = self.predictor['X'].predict(cam_coordinate)
        # robot_coordinate_y = self.predictor['Y'].predict(cam_coordinate)
        # robot_coordinate_z = self.predictor['Z'].predict(cam_coordinate)
        # robot_coordinate = np.array([robot_coordinate_x,robot_coordinate_y,robot_coordinate_z]).ravel()/1000

        robot_coordinate = self.predictor['Linear'].predict(cam_coordinate).ravel()/1000
        np.savetxt('predict_test.txt',robot_coordinate)
        return robot_coordinate.tolist()

    def cali_joint(self, id, cam_coordinate):
        cali_deg = self.predictor['J1'].predict(cam_coordinate)[0]
        solution = self.get_joints_list()
        solution[id-1] = int(cali_deg)
        self.move_servo_to(solution)
        self.update_joints_list()
        return 
    
    def set_torque(self):
        self.Arm.Arm_Button_Mode(0)
        return
    
    def off_torque(self):
        self.Arm.Arm_Button_Mode(1)
        return

    @staticmethod
    def radius_to_servo_degree(value_in_radius, offset=0):
        """Convert radius to degree."""
        value_in_degree = np.rad2deg(value_in_radius)
        value_in_degree_robot = value_in_degree+90  # the initial state of servo is 90
        value_in_degree_robot[3]+=offset
        return value_in_degree_robot

    @staticmethod
    def check_limitation(planned_joints_list):
        """Check if the robot's movement violates any limitations."""
        # To be implemented
        flag = 0
        for i, joint in enumerate(planned_joints_list):
            if i==0 and -5<=joint<=185:
                flag = 1 
            elif i!=0 and -45<=joint<=240:
                flag =  1
            else:
                flag =0
                # print()
                break
        return flag
            
    @staticmethod
    def kdl_to_np(M):
        result = None
        if isinstance(M, type(kdl.Rotation())):
            result = np.zeros([3,3])
            for i in range(3):
                for j in range(3):
                    result[i,j] = M[i,j]
        elif isinstance(M, type(kdl.Frame())):
            result = np.eye(4)
            R = M.M
            p = M.p
            for i in range(3):
                for j in range(3):
                    result[i,j] = M[i,j]
            for i, k in enumerate(p):
                result[i,3] = k
        else:
            result = []
            for v in M:
                result.append(v)
            result = np.array(result)
        return result


