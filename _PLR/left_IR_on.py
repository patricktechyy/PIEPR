import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BCM)

gpio.setup(2, gpio.OUT)
gpio.setup(3, gpio.OUT)
gpio.setup(4, gpio.OUT)

on = gpio.HIGH
off = gpio.LOW

gpio.output(2, on)
gpio.output(3, on)
gpio.output(4, on)

print("Left IR Illumination ON")
