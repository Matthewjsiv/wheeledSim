import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time

from ackermann_msgs.msg import AckermannDriveStamped
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

if __name__ == '__main__':
    from kbm_planner import KBMMPCPlanner

    env = rosSimController('../../configurations/testPlanner.yaml', render=True)

    #Get map resolution params
    hmap_params = next(x.senseParams for x in env.sensors if x.topic == 'heightmap')

    kbm = KBMKinematics({'L':0.9}, reverse_steer=True)
    throttle_tf = torch.load('f1p0_throttle.pt')
    steer_tf = torch.load('f1p0_steer.pt')
    dynamic_kbm = KBMDynamics(kbm, throttle_tf, steer_tf)

    primitives = get_primitives(throttle_n=1, steer_n=7, T=15)

    planner = KBMMPCPlanner(dynamic_kbm, primitives, hmap_params)

#    planner.position = torch.tensor([0., 0., np.pi/2])
#    planner.goal = torch.tensor([5.0, 0.0])
#    planner.get_relative_goal()

    freq = 10
    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(freq)

    #Simulator
    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    heightmap_pub = rospy.Publisher("/heightmap", GridMap, queue_size=1)

    #Planner
    heightmap_sub = rospy.Subscriber("/heightmap", GridMap, planner.handle_heightmap)
    odom_sub = rospy.Subscriber("/odom", Odometry, planner.handle_odom)
    goal_sub = rospy.Subscriber("/goal", PoseStamped, planner.handle_goal)
    cmd_pub = rospy.Publisher("/cmd", AckermannDriveStamped, queue_size=1)

    T = 50
    cmd_buf = torch.zeros(T, 2)
    pred_state_buf = torch.zeros(T, 2)
    gt_state_buf = torch.zeros(T, 2)
    pred_traj_buf = torch.zeros(T, 3)
    gt_traj_buf = torch.zeros(T, 2)
    time_buf = []

    fig, axs = planner.render()
    plt.show(block=False)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()
        ctrl = torch.tensor(env.curr_drive).float()
        pred = dynamic_kbm.forward(odom_to_state(state), ctrl, dt=1./freq)

        odom_pub.publish(state)
        heightmap_pub.publish(sensing['heightmap'])
        cmd_pub.publish(torch_to_cmd(planner.best_seq[0]))

        for ax in axs:
            ax.cla()

        ts = time.time()
        planner.plan()
        te = time.time()
        time_buf.append(te-ts)
        if len(time_buf) > T:
            time_buf = time_buf[1:]

#        torch.set_printoptions(sci_mode=False, precision=4)
#        print('Current Position = {}'.format(planner.position))
#        print('Current Goal =     {}'.format(planner.goal))
#        print('Relative Goal =    {}'.format(planner.relative_goal))

        print('Plan time: {:.4f}s'.format(sum(time_buf)/len(time_buf)))
        print('Act = {}'.format(planner.best_seq[0]))

        planner.render(fig, axs)

        plt.pause(1e-2)

        rate.sleep()
