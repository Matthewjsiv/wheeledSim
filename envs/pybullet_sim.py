import gym
import numpy as np
import pybullet
import yaml

from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController


class WheeledSimEnv:
    """
    Wrapper class to make a gym environment from pybullet sim
    """

    def __init__(self, config_file, use_images=False, T=-1, render=True):

        stream = open(config_file, 'r')
        config = yaml.load(stream, Loader=yaml.FullLoader)

        self.client = pybullet.connect(pybullet.GUI) if render else pybullet.connect(pybullet.DIRECT)
        self.robot = Clifford(params=config['cliffordParams'], physicsClientId=self.client)

        self.env = simController(self.robot, self.client, config['simulationParams'], config['senseParams'],
                                 config['terrainMapParams'], config['terrainParams'], config['explorationParams'])

        self.T = T  # max steps allowed
        self.nsteps = 0  # number of steps taken
        self.use_images = use_images

    @property
    def observation_space(self):
        # observation takes form (x,y,z) position, (x,y,z,w) quaternion orientation, velocity, joint state
        state_space = gym.spaces.Box(low=np.ones(13) * -float('inf'), high=np.ones(13) + float('inf'))
        if not self.use_images:
            return state_space
        else:
            image_space = gym.spaces.Box(low=np.ones(self.env.senseParams['senseResolution']) * -float('inf'),
                                         high=np.ones(self.env.senseParams['senseResolution']) * float('inf'))
            return gym.spaces.Dict({'state': state_space, 'image': image_space})

    @property
    def action_space(self):
        # action takes form [throttle, steering]
        return gym.spaces.Box(low=-np.ones(2), high=np.ones(2))

    def reset(self):
        # reset environment
        self.env.newTerrain()
        self.nsteps = 0
        self.env.controlLoopStep([0., 0.])
        self.env.resetRobot()

        # get new observation
        pose = self.env.robot.getPositionOrientation()
        vel = self.robot.getBaseVelocity_body()
        joints = self.robot.measureJoints()
        if self.env.senseParams['recordJointStates']:
            obs = list(pose[0]) + list(pose[1]) + vel[:] + joints[:]
        else:
            obs = list(pose[0]) + list(pose[1]) + vel[:]

        if self.use_images:
            hmap = self.env.sensing(pose, senseType=0)
            return {"state": np.array(obs), "image": hmap}

        else:
            return np.array(obs)

    def step(self, action):
        # get state action, next state, and boolean terminal state from simulation
        sa, s, sim_t = self.env.controlLoopStep(action)  # image is sa[1]
        obs = {"state": np.array(s)[0], "image": sa[1]} if self.use_images else np.array(s)[0]

        # increment number of steps and set terminal state if reached max steps
        self.nsteps += 1
        timeout = self.get_terminal()
        reward = self.get_reward(obs)

        return obs, reward, sim_t or timeout, {}

    def get_reward(self, obs):
        # can be updated with a reward function in future work
        return 0

    def get_terminal(self):
        return (self.T > 0) and (self.nsteps >= self.T)


if __name__ == '__main__':

    """load environment"""
    config_file = "configurations/cliffordExampleParams.yaml"
    env = WheeledSimEnv(config_file, T=50)

    # run simulation 5 times
    for _ in range(5):
        terminal = False
        time = 0
        while not terminal:
            time += 1
            a = env.env.randomDriveAction()
            obs, reward, terminal, i = env.step(a)
            print('STATE = {}, ACTION = {}, t = {}'.format(obs, a, time))

        env.reset()
