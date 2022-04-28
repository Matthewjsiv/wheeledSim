import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from common_msgs.msg import AckermannDriveArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.msg import GridMap
from rosgraph_msgs.msg import Clock

from wheeledSim.rosSimController import rosSimController

if __name__ == '__main__':
    from wheeledRobots.clifford.cliffordRobot import Clifford

    from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor
    from wheeledSim.sensors.local_heightmap_sensor import LocalHeightmapSensor
    from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
    from wheeledSim.sensors.lidar_sensor import LidarSensor

    rate = 10
    dt = 1./rate
    curr_time = 0.

    env = rosSimController('../../configurations/racetrack.yaml', render=True)

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(10)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)

    clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    cam_pub = rospy.Publisher("/front_camera", Image, queue_size=1)

    while not rospy.is_shutdown():
        t1 = time.time()
        env.step()
        t2 = time.time()
        state, sensing = env.get_sensing()
        img = sensing['front_camera']
        t3 = time.time()

        odom_pub.publish(state)
        cam_pub.publish(img)

        print('STEP: {:.6f}, SENSE: {:.6f}'.format(t2-t1, t3-t2))

        rate.sleep()
