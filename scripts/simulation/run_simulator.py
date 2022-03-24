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

    rate = 5
    dt = 1./rate
    curr_time = 0.

    env = rosSimController('../../configurations/heightmap_frontcam.yaml', render=True)

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(5)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)

    clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    map_pub = rospy.Publisher("/local_map", GridMap, queue_size=1)
    cam_pub = rospy.Publisher("/front_camera", Image, queue_size=1)

    while not rospy.is_shutdown():
        t1 = time.time()
        env.step()
        t2 = time.time()
        state, sensing = env.get_sensing()
        t3 = time.time()
        heightmap = sensing['heightmap']
        img = sensing['front_camera']

        odom_pub.publish(state)
        map_pub.publish(heightmap)
        cam_pub.publish(img)

        print('STEP: {:.6f}, SENSE: {:.6f}'.format(t2-t1, t3-t2))

        curr_time += dt
        clock_msg = Clock()
        clock_msg.clock = rospy.Time.from_sec(curr_time)
        clock_pub.publish(clock_msg)

#        rate.sleep()
