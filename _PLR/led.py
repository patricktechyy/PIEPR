from time import sleep
import RPi.GPIO as GPIO

BLUE_PIN = 19
RED_PIN = 13

PWM_HZ = 500 #in hertz HZ

BLUE_DUTY = 15.5 #equalized blue pwm to match red's irradiance
RED_DUTY = 100

PRE_STIM = 3 #pre-stimulus baseline
STIM_DUR = 1
REST_1 = 60 #in seconds
REST_2 = 60

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(BLUE_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)

blue_pwm = GPIO.PWM(BLUE_PIN, PWM_HZ)
red_pwm = GPIO.PWM(RED_PIN, PWM_HZ)

try:
    sleep(PRE_STIM)

    print("Blue LED ON")
    blue_pwm.start(BLUE_DUTY)
    sleep(STIM_DUR)
    blue_pwm.stop()
    print("Blue LED OFF")

    sleep(REST_1)

    print("Red LED ON")
    red_pwm.start(RED_DUTY)
    sleep(STIM_DUR)
    red_pwm.stop()
    print("Red LED OFF")

    sleep(REST_2)

finally:
    # cleanup only the pins that we used and not globally
    GPIO.cleanup([BLUE_PIN, RED_PIN])
