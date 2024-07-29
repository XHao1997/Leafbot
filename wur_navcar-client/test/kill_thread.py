import threading
import time


class MyThread(threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()
        self.stop_event = threading.Event()

    def run(self):
        # 清空设置
        self.stop_event.clear()
        while not self.stop_event.is_set():
            print("Thread is running...")
            time.sleep(1)

    def stop(self):
        self.stop_event.set()

    # 创建线程


thread = MyThread()
thread.start()

# 在某个时间点，停止线程  
time.sleep(5)
thread.stop()
