import signal
import sys
import time
import RPi.GPIO as GPIO

PIN = 12
PWM_HZ = 500 #in hertz HZ
DUTY = 20  #just a rough low value

running = True

def handle_signal(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN, GPIO.OUT)

pwm = GPIO.PWM(PIN, PWM_HZ)

try:
    pwm.start(DUTY)
    print("Dim LED ON")
    while running:
        time.sleep(0.2)
finally:
    try:
        pwm.stop()
    except Exception:
        pass
    # cleanup only our pin, do NOT global ccleanup or else there might be conflicts
    GPIO.cleanup(PIN)
    print("Dim LED OFF")
