#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
import math 
import time
import numpy as np
from tabulate import tabulate

class JoyToSpeed(object):
    def __init__(self):
        print('Node_init..')
        rospy.Subscriber('joy', Joy, self.handle_joy)
        self.cmdPub = rospy.Publisher('/cmd',AckermannDriveStamped,queue_size=10)

        #The controller can change these
        self.params = {
            'K_throttle':1.0,
            'K_steer':0.37,
            'noise_throttle':0.0,
            'noise_steer':0.0, 
        }

        #The controller can't change these
        self.throttle_step = 0.1
        self.steer_step = 0.1
        self.noise_step = 0.01
        self.K_throttle_min = 0.0#0.5
        self.K_throttle_max = 1.0
        self.K_steer_min = 0.
        self.K_steer_max = 1.0#0.5
        self.noise_throttle_min = 0.
        self.noise_throttle_max = 0.2
        self.noise_steer_min = 0.
        self.noise_steer_max = 0.1
        self.k_trim = 0.1
        self.noise_threshold = 0.1
        self.noise_limit = 0.1

        self.cmd = AckermannDriveStamped()
        self.joy_msg = Joy()
        self.joy_msg.axes = [0] * 6
        self.joy_msg.buttons = [0]  * 14

    def update_params(self, msg):
        #Noise values
        if msg.buttons[0]:
            self.params['noise_throttle'] = min(max(self.params['noise_throttle'] + self.noise_step, self.noise_throttle_min), self.noise_throttle_max)
        if msg.buttons[1]:
            self.params['noise_throttle'] = min(max(self.params['noise_throttle'] - self.noise_step, self.noise_throttle_min), self.noise_throttle_max)
        if msg.buttons[2]:
            self.params['noise_steer'] = min(max(self.params['noise_steer'] - self.noise_step, self.noise_steer_min), self.noise_steer_max)
        if msg.buttons[3]:
            self.params['noise_steer'] = min(max(self.params['noise_steer'] + self.noise_step, self.noise_steer_min), self.noise_steer_max)

        #Throttle/steer values
        if msg.buttons[4]:
            self.params['K_throttle'] = min(max(self.params['K_throttle'] + self.throttle_step, self.K_throttle_min), self.K_throttle_max)
        if msg.buttons[5]:
            self.params['K_steer'] = min(max(self.params['K_steer'] + self.steer_step, self.K_steer_min), self.K_steer_max)
        if msg.buttons[6]:
            self.params['K_throttle'] = min(max(self.params['K_throttle'] - self.throttle_step, self.K_throttle_min), self.K_throttle_max)
        if msg.buttons[7]:
            self.params['K_steer'] = min(max(self.params['K_steer'] - self.steer_step, self.K_steer_min), self.K_steer_max)

        #Throttle/steer trim (these are on an axis)
#        self.params['K_throttle'] = min(max(self.params['K_throttle'] + self.throttle_step * self.k_trim * msg.axes[5], self.K_throttle_min), self.K_throttle_max)
#        self.params['K_steer'] = min(max(self.params['K_steer'] - self.steer_step * self.k_trim * msg.axes[4], self.K_steer_min), self.K_steer_max)

    def update_cmd(self):
        throttle_noise = np.random.randn() * self.params['noise_throttle']
        steer_noise = np.random.randn() * self.params['noise_steer']
        throttle_noise = min(max(throttle_noise, -self.noise_limit), self.noise_limit)
        steer_noise = min(max(steer_noise, -self.noise_limit), self.noise_limit)

        self.cmd.header.stamp = rospy.Time.now()
        self.cmd.drive.speed = self.joy_msg.axes[1] * self.params['K_throttle']
        self.cmd.drive.steering_angle = self.joy_msg.axes[3] * self.params['K_steer']

        if abs(self.cmd.drive.speed) > self.noise_threshold:
            self.cmd.drive.speed += throttle_noise
            self.cmd.drive.steering_angle += steer_noise

    def handle_joy(self, msg):
        self.update_params(msg)
        self.joy_msg = msg

    def update(self):
        print(self.joy_msg)
        self.cmdPub.publish(self.cmd)
        self.update_cmd()
        print(tabulate([[k, self.params[k]] for k in self.params.keys()], tablefmt='psql'))

if __name__ == "__main__":
    print('start..')
    rospy.init_node('joy_to_speed', anonymous=True)
    print('init..')
    joy2speed = JoyToSpeed()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        joy2speed.update()
        rate.sleep()
