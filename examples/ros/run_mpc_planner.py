import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time
import yaml

from ackermann_msgs.msg import AckermannDriveStamped
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

from kbm_planner import KBMMPCPlanner, KBMAstarPlanner

def get_cmds(x):
    """
    Get cmds (as an array of Ackermann motions) from a torch tensor of throttle, steer
    """
    msg = AckermannDriveArray()
    drives = [torch_to_cmd(y).drive for y in x]
    msg.drives = drives
    msg.header.stamp = rospy.Time.now()
    return msg

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

def get_primitives(throttle_n, steer_n, T=10, reverse=False):
    seqs = torch.zeros(throttle_n, steer_n, T, 2)
    if reverse:
        throttles = torch.linspace(-1, 1, throttle_n)
    else:
        throttles = torch.linspace(0., 1., throttle_n + 1)[1:]
    steers = torch.linspace(-1., 1., steer_n)

    for ti, t in enumerate(throttles):
        for si, s in enumerate(steers):
            seqs[ti, si, :, 0] = t
            seqs[ti, si, :, 1] = s
    seqs = seqs.view(-1, T, 2)
    seqs = torch.cat([seqs, torch.zeros_like(seqs[[0]])], dim=0)
    return seqs

if __name__ == '__main__':

    params = yaml.safe_load(open('../../configurations/testPlanner.yaml', 'r'))

    #Get map resolution params
    hmap_params = next(x['params'] for x in params['sensors'].values() if x['topic'] == 'heightmap')
    controller_freq = 10

    kbm = KBMKinematics({'L':0.9}, reverse_steer=True)
    throttle_tf = torch.load('f1p0_throttle.pt')
    steer_tf = torch.load('f1p0_steer.pt')
    dynamic_kbm = KBMDynamics(kbm, throttle_tf, steer_tf)

#    primitives = get_primitives(throttle_n=2, steer_n=5, T=20, reverse=False)
#    planner = KBMMPCPlanner(dynamic_kbm, primitives, hmap_params, dt=1./controller_freq, relative_goal=True)

    primitives = get_primitives(throttle_n=2, steer_n=5, T=5, reverse=False)
    planner = KBMAstarPlanner(dynamic_kbm, primitives, hmap_params, dt=1./controller_freq, max_itrs=10, relative_goal=True)

    freq = 3
    rospy.init_node("mpc_planner")
    rate = rospy.Rate(freq)

    #Planner
    heightmap_sub = rospy.Subscriber("/local_height_map", GridMap, planner.handle_heightmap)
#    heightmap_sub = rospy.Subscriber("/heightmap", GridMap, planner.handle_heightmap)
    odom_sub = rospy.Subscriber("/tartanvo_odom", Odometry, planner.handle_odom)
    goal_sub = rospy.Subscriber("/goal", PoseStamped, planner.handle_goal)
    plan_pub = rospy.Publisher("/plan", AckermannDriveArray, queue_size=1)

    fig, axs = planner.render()
    plt.show(block=False)

    T = 50
    time_buf = []

    while not rospy.is_shutdown():
        msg = get_cmds(planner.best_seq)
        plan_pub.publish(msg)

        for ax in axs:
            ax.cla()

        ts = time.time()
        planner.plan()
        te = time.time()
        time_buf.append(te-ts)
        if len(time_buf) > T:
            time_buf = time_buf[1:]

        torch.set_printoptions(sci_mode=False, precision=4)
        print('Current Position = {}'.format(planner.position))
        print('Current Goal =     {}'.format(planner.goal))
        print('Relative Goal =    {}'.format(planner.relative_goal))

        print('Plan time: {:.4f}s'.format(sum(time_buf)/len(time_buf)))
        print('Act = {}'.format(planner.best_seq[0]))

        planner.render(fig, axs)

        plt.pause(1e-2)

        rate.sleep()
