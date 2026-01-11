import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BCM)

gpio.setup(15, gpio.OUT)

gpio.output(15, gpio.HIGH)
sleep(0.5)
gpio.output(15, gpio.LOW)

sleep(300)

gpio.output(15, gpio.HIGH)
sleep(0.5)
gpio.output(15,gpio.LOW)

gpio.cleanup()
