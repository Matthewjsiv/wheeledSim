import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import time
import argparse

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from wheeledSim.rosSimController import rosSimController

def cvt_ShockTravelArray(msg):
    data = torch.zeros(len(msg.shock_travels), 4)
    for i, shock_travel in enumerate(msg.shock_travels):
        data[i] = torch.tensor([shock_travel.front_left, shock_travel.front_right, shock_travel.rear_left, shock_travel.rear_right]).float()

    return data

def cvt_ImuArray(msg):
    data = torch.zeros(len(msg.imus), 6)
    for i, imu in enumerate(msg.imus):
        data[i] = torch.tensor([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z, imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z]).float()

    return data

def get_fft_histogram(data):
    """
    Get the coeffs of fft for time series batched as [batch x time]
    """
    N = data.shape[0]
    fhat = torch.fft.fft(data, n=N, dim=0)
    PSD = fhat * torch.conj(fhat) / N
    return PSD[:int(N/2)].real

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to envspec')
    args = parser.parse_args()

    env = rosSimController(args.config, render=True)
    #Get map resolution params

    freq = 10
    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(freq)

    #Simulator
    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)

    T = 50
    cmd_buf = torch.zeros(T, 2)
    pred_state_buf = torch.zeros(T, 2)
    gt_state_buf = torch.zeros(T, 2)
    pred_traj_buf = torch.zeros(T, 3)
    gt_traj_buf = torch.zeros(T, 2)
    time_buf = []

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    plt.show(block=False)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()

        shock_data = cvt_ShockTravelArray(sensing['shock_travel'])
        imu_data = cvt_ImuArray(sensing['imu'])

        shock_fft = get_fft_histogram(shock_data)
        imu_fft = get_fft_histogram(imu_data)

        for ax in axs.flatten():
            ax.cla()

        for i, l in enumerate(['fl', 'fr', 'rl', 'rr']):
#            axs[0, 0].plot(shock_fft[1:, i]/shock_fft[1:, i].sum(), label=l)
            axs[0, 0].bar(torch.arange(shock_fft.shape[0]-1), shock_fft.sum(dim=-1)[1:])
            axs[0, 1].plot(shock_data[:, i], label=l)

        for i, l in enumerate(['wx', 'wy', 'wz', 'ax', 'ay', 'az']):
#            axs[1, 0].plot(imu_fft[1:, i]/imu_fft[1:, i].sum(), label=l)
            axs[1, 0].bar(torch.arange(imu_fft.shape[0]-1), imu_fft.sum(dim=-1)[1:])
            axs[1, 1].plot(imu_data[:, i], label=l)

        axs[0, 0].set_title('Shock FFT')
        axs[0, 1].set_title('Shock Data')
        axs[1, 0].set_title('IMU FFT')
        axs[1, 1].set_title('IMU Data')

        for ax in axs.flatten():
            ax.legend()

        plt.pause(1e-2)

        rate.sleep()
