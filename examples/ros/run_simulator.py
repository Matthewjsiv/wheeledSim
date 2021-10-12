import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from common_msgs.msg import AckermannDriveArray
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped

from wheeledSim.rosSimController import rosSimController

if __name__ == '__main__':
    from grid_map_msgs.msg import GridMap

    from wheeledRobots.clifford.cliffordRobot import Clifford

    from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor
    from wheeledSim.sensors.local_heightmap_sensor import LocalHeightmapSensor
    from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
    from wheeledSim.sensors.lidar_sensor import LidarSensor

    env = rosSimController('../../configurations/latentSensorParams.yaml', render=True)

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(10)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    map_pub = rospy.Publisher("/local_map", GridMap, queue_size=1)
    cam_pub = rospy.Publisher("/front_camera", GridMap, queue_size=1)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()
        heightmap = sensing['heightmap']
        img = sensing['front_camera']
        odom_pub.publish(state)
        map_pub.publish(heightmap)
        cam_pub.publish(img)
        rate.sleep()
