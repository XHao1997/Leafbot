import os
import socketserver
from module.robot import Robot
import numpy as np
import time
import copy
import socket
import pickle

ACTION_DONE = 10
IN_PROGRESS = 20
MOVE_ARM_FOR_CALI_TASK = 3
MOVE_ARM_RANDOM_TASK = 4
MOVE_ARM_FOR_GRAP_TASK = 5
COLLECT_JOINT1_FOR_NN = 6
PICK_LEAF = 7
CONTINUE = 2

class RobotServer(socketserver.BaseRequestHandler):
    """Handles incoming requests from clients via a socket server."""
    
    def __init__(self, request, client_address, server):
        """Initialize the RobotServer and the robot."""
        super().__init__(request, client_address, server)

    def handle(self):
        """Handles client connections."""
        conn = self.request
        conn.sendall('Welcome to the socketserver server!'.encode())
        robot = Robot()
        robot.initialize_robot()
        while True:
            data = conn.recv(1024).decode()
            if data == "0":
                print("Disconnecting from %s" % (self.client_address,))
                conn.shutdown(socket.SHUT_RDWR)  # Shut down the connection socket
                conn.close()  # Close the connection socket
                break
            elif data == str(MOVE_ARM_FOR_CALI_TASK):
                actually_joints_list = []
                for i, position in enumerate(robot.generate_nn_positions()):
                    if i==0:
                        speed = 1000
                    else:
                        speed = 750
                    robot.move_servo_to(position,speed) 
                    time.sleep(10)
                    robot.update_joints_list() 
                    current_joints_list = copy.deepcopy(robot.get_joints_list())
                    actually_joints_list.append(current_joints_list)
                    msg = str("capture")
                    conn.sendall(msg.encode())
                    time.sleep(5)
                np.save('joints_eye_to_hand', actually_joints_list)
                msg=str(ACTION_DONE)
                conn.sendall(msg.encode())
            elif data == str(COLLECT_JOINT1_FOR_NN):
                actually_joints_list = []
                capture_done = None
                n_times = int(input('input n_times'))
                for i in range(n_times):
                    if i==0:
                        print(i)
                        robot.move_servo_to(next(robot.generate_nn_positions()))
                        robot.update_joints_list() 
                        time.sleep(10)
                        msg=str('capture')
                    else:
                        capture_done = int(conn.recv(1024).decode())==CONTINUE
                        if capture_done:
                            print(i)
                            robot.move_servo_to(next(robot.generate_nn_positions()))
                            robot.update_joints_list() 
                            time.sleep(10)
                            msg=str('capture')
                    current_joints_list = copy.deepcopy(robot.get_joints_list())
                    actually_joints_list.append(current_joints_list)
                    conn.sendall(msg.encode())
                    time.sleep(20)
                np.save('joint1_nn', actually_joints_list)
                msg=str(ACTION_DONE)
                conn.sendall(msg.encode())
            elif data == str(PICK_LEAF):
                robot.move_servo_to(next(robot.generate_nn_positions()))
                robot.update_joints_list() 
                time.sleep(1)
                msg=str('capture')
                actually_joints_list = []
                capture_done = None
                msg='capture'
                conn.sendall(msg.encode())
                data = conn.recv(1024)
                data = pickle.loads(data)
                data = np.asarray(data,dtype=np.int16).reshape(1,-1)
                robot_xyz = robot.coodinate_cam2robot(data)
                desired_position = {'XYZ':robot_xyz,
                    'RPY':[-np.pi/2,0,0]}

                robot.move_to_zero_position()
                robot.move_to(desired_position)
                robot.cali_joint(id=1,cam_coordinate=data)
                msg=str(ACTION_DONE)
                conn.sendall(msg.encode())
            else:
                print("Client at %s sent message: %s" % (self.client_address, data))
                conn.sendall(('Received your message <%s>' % data).encode())
if __name__ == '__main__':
    # Specify the IP address and port to listen on
    ip_address = '192.168.101.11'
    port = 4000
    # Get the directory of the main Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the current working directory to the script directory
    os.chdir(script_directory)
    # Create the server
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((ip_address, port), RobotServer)
    
    print("Starting the socketserver server!")
    
    try:
        # Start the server
        server.serve_forever()
    except KeyboardInterrupt:
        print("Closing the server...")
        # Close the server when interrupted by KeyboardInterrupt
        
        server.server_close()
