import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BCM)

gpio.setwarnings(False)

gpio.setup(2, gpio.OUT)
gpio.setup(3, gpio.OUT)
gpio.setup(4, gpio.OUT)

on = 1
off = 0

gpio.output(2, off)
gpio.output(3, off)
gpio.output(4, off)

print("Left IR Illumination OFF")

gpio.cleanup()
print("Left IR GPIO cleaned")
