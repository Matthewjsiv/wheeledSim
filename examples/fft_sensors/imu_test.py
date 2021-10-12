import numpy as np
import matplotlib.pyplot as plt
import argparse

from wheeledSim.envs.pybullet_sim import WheeledSimEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to envspec')
    args = parser.parse_args()

    env = WheeledSimEnv(args.config)

    import pdb;pdb.set_trace()

    obs = env.reset()
    nobs, r, t, i = env.step([1.0, 0.0])

    vels = nobs['imu']

    #Euler-integrate the vels
