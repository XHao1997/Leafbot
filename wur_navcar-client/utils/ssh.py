import warnings

warnings.filterwarnings('ignore')
import paramiko
import time


def ssh_connect(_host, _username, _password):
    try:
        _ssh_fd = paramiko.SSHClient()
        _ssh_fd.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        _ssh_fd.connect(_host, username=_username, password=_password, timeout=10)
    except Exception:
        print('Authorization Failed!Please check the username,password or your device is connected to the Internet.')
        exit()
    return _ssh_fd


def ssh_exec_cmd(_ssh_fd, _cmd):
    return _ssh_fd.exec_command(_cmd)


def ssh_close(_ssh_fd):
    _ssh_fd.close()


def print_ssh_exec_cmd_return(_ssh_fd, _cmd):
    ssh_exec_cmd(_ssh_fd, _cmd)


def run_remote_stream():
    sshd2 = ssh_connect('192.168.101.11', 'dofbot', 'yahboom')
    print_ssh_exec_cmd_return(sshd2, 'pkill python')
    print_ssh_exec_cmd_return(sshd2, 'cd Dofbot/robotarm_sim-main/\n/usr/bin/python3 arm_sub.py\n')

    # sshd = ssh_connect('192.168.101.12', 'hao', '333338')
    # print_ssh_exec_cmd_return(sshd, 'sudo pkill python')
    # ssh_exec_cmd(sshd, r'/home/hao/Desktop/wur_navcar/remote_car/bin/python '
    #                    r'/home/hao/Desktop/wur_navcar/test/test_zmq.py')

    return
