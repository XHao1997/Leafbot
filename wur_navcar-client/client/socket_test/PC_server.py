import socket
import time
import sys
#RPi's IP
SERVER_IP = "192.168.1.101"
SERVER_PORT = 8888

print("Starting socket: TCP...")
server_addr = (SERVER_IP, SERVER_PORT)
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Connecting to server @ %s:%d..." %(SERVER_IP, SERVER_PORT))
socket_tcp.connect((SERVER_IP,8888))

while True:
    try:
        data = socket_tcp.recv(512)

        print("Received: %s" % data)
        command=input()
        socket_tcp.send(command.encode())
        time.sleep(1)
        continue
    except Exception:
        socket_tcp.close()
        socket_tcp=None
        sys.exit(1)
