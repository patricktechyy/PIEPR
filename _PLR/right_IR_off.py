import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BCM)

gpio.setwarnings(False)

gpio.setup(23, gpio.OUT)
gpio.setup(24, gpio.OUT)
gpio.setup(25, gpio.OUT)

on = 1
off = 0

gpio.output(23, off)
gpio.output(24, off)
gpio.output(25, off)

print("Right IR Illumination OFF")

gpio.cleanup()
print("Right IR GPIO cleaned")
