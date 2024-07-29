# -*- coding: utf-8 -*-
import RPi.GPIO as GPIO
import time
def sensor_detect():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    GPIO.output(18, GPIO.HIGH)          
    time.sleep(2)
    GPIO.output(18, GPIO.LOW)
    GPIO.cleanup()
