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
from wheeledSim.models.kbm_dynamics import KBMDynamics
from wheeledSim.models.kinematic_bicycle_model import KBMKinematics
from wheeledSim.models.transfer_functions import ARXTransferFunction
from wheeledSim.heightmap import HeightMap
from wheeledSim.util import yaw_to_quat, quat_to_yaw

def torch_to_cmd(x):
    cmd = AckermannDriveStamped()
    cmd.header.stamp = rospy.Time.now()
    cmd.drive.speed = x[0].item()
    cmd.drive.steering_angle = x[1].item()
    return cmd

def odom_to_state(odom):
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    yaw = quat_to_yaw(odom.pose.pose.orientation)
    return torch.tensor([x, y, yaw]).float()

def state_to_odom(state, vel):
    x, y, yaw = state[:3]
    qx, qy, qz, qw = yaw_to_quat(yaw)
    msg = Odometry()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = vel

    msg.header.stamp = rospy.Time.now()
    return msg

def heightmap_to_numpy(hmap_msg):
    """
    Decode heightmap msg back to numpy
    """
    height_idx = hmap_msg.layers.index('height')
    data = np.array(hmap_msg.data[height_idx].data)
    rowwidth = hmap_msg.data[height_idx].layout.dim[1].size
    hmap = data.reshape(-1, rowwidth)
    return hmap

def get_primitives(throttle_n, steer_n, T=10):
    seqs = torch.zeros(throttle_n, steer_n, T, 2)
    throttles = torch.linspace(0., 1., throttle_n + 1)[1:]
    steers = torch.linspace(-1., 1., steer_n)

    for ti, t in enumerate(throttles):
        for si, s in enumerate(steers):
            seqs[ti, si, :, 0] = t
            seqs[ti, si, :, 1] = s
    seqs = seqs.view(-1, T, 2)
    seqs = torch.cat([seqs, torch.zeros_like(seqs[[0]])], dim=0)
    return seqs

class Controller:
    """
    Actually execute the commands from the planner.
    """
    def __init__(self):
        self.plan = [AckermannDrive()]
        self.t = 0

    def handle_plan(self, msg):
        """
        load an array of AckermannDrives
        """
        self.plan = msg.drives
        self.t = 0

    def get_cmd(self):
        cmd = AckermannDriveStamped()
        if self.t < len(self.plan):
            cmd.drive = self.plan[self.t]
        cmd.header.stamp = rospy.Time.now()
        self.t += 1

        return cmd

if __name__ == '__main__':
    from kbm_planner import KBMMPCPlanner

    env = rosSimController('../../configurations/testPlanner.yaml', render=True)
    controller = Controller()

    #Get map resolution params

    freq = 10
    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(freq)

    #Simulator
    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    heightmap_pub = rospy.Publisher("/heightmap", GridMap, queue_size=1)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()

        odom_pub.publish(state)
        heightmap_pub.publish(sensing['heightmap'])

        rate.sleep()
