import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from wheeledSim.rosSimController import rosSimController
from wheeledSim.models.kbm_dynamics import KBMDynamics
from wheeledSim.models.kinematic_bicycle_model import KBMKinematics
from wheeledSim.models.transfer_functions import ARXTransferFunction
from wheeledSim.util import yaw_to_quat

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

if __name__ == '__main__':
    env = rosSimController('../../configurations/testPlanner.yaml', render=True)
    kbm = KBMKinematics({'L':0.9}, reverse_steer=True)
    throttle_tf = torch.load('f1p0_throttle.pt')
    steer_tf = torch.load('f1p0_steer.pt')

    dynamic_kbm = KBMDynamics(kbm, throttle_tf, steer_tf)

    freq = 10
    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(freq)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    pred_odom_pub = rospy.Publisher("/predicted_odom", Odometry, queue_size=1)

    T = 50
    cmd_buf = torch.zeros(T, 2)
    pred_state_buf = torch.zeros(T, 2)
    gt_state_buf = torch.zeros(T, 2)
    pred_traj_buf = torch.zeros(T, 3)
    gt_traj_buf = torch.zeros(T, 2)


    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    plt.show(block=False)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()
        ctrl = torch.tensor(env.curr_drive)
        pred = dynamic_kbm.forward(pred_traj_buf[-1], ctrl, dt=1./freq)

        odom_pub.publish(state)
        pred_odom_pub.publish(state_to_odom(pred, dynamic_kbm.prev_vel))

        #Handle data and plotting
        cmd_buf = torch.cat([cmd_buf[1:], ctrl.unsqueeze(0)], dim=0)
        pred_state_buf = torch.cat([pred_state_buf[1:], torch.stack([dynamic_kbm.prev_vel, dynamic_kbm.prev_yaw]).unsqueeze(0)], dim=0)
        gt_state_buf = torch.cat([gt_state_buf[1:], torch.tensor([state.twist.twist.linear.x, pybullet.getJointState(env.robot.cliffordID, env.robot.jointNameToID['axle2frwheel'])[0]]).unsqueeze(0)])
        gt_traj_buf = torch.cat([gt_traj_buf[1:], torch.tensor([state.pose.pose.position.x, state.pose.pose.position.y]).unsqueeze(0)], dim=0)
        pred_traj_buf = torch.cat([pred_traj_buf[1:], pred.unsqueeze(0)], dim=0)

        for ax in axs:
            ax.cla()

        axs[0].plot(pred_traj_buf[:, 0], pred_traj_buf[:, 1], c='r', label='Predicted Traj')
        axs[0].plot(gt_traj_buf[:, 0], gt_traj_buf[:, 1], c='b', label='GT Traj')
        axs[0].plot(pred_traj_buf[-1, 0], pred_traj_buf[-1, 1], c='r', marker='x', label='Predicted Traj')
        axs[0].plot(gt_traj_buf[-1, 0], gt_traj_buf[-1, 1], c='b', marker='x', label='GT Traj')

        axs[1].plot(pred_state_buf[:, 0], label='Predicted velocity')
        axs[1].plot(pred_state_buf[:, 1], label='Predicted steer')
        axs[1].plot(gt_state_buf[:, 0], label='GT velocity')
        axs[1].plot(gt_state_buf[:, 1], label='GT steer')

        axs[2].plot(cmd_buf[:, 0], label='Throttle')
        axs[2].plot(cmd_buf[:, 1], label='Steer')
        axs[2].set_ylim(-1.1, 1.1)

        for ax in axs:
            ax.legend()

        plt.pause(1e-2)

        rate.sleep()
