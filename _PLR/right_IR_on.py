import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BCM)

gpio.setup(23, gpio.OUT)
gpio.setup(24, gpio.OUT)
gpio.setup(25, gpio.OUT)

on = gpio.HIGH
off = gpio.LOW

gpio.output(23, on)
gpio.output(24, on)
gpio.output(25, on)

print("Right IR Illumination ON")
