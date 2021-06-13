import gym
import numpy as np
import pybullet
import yaml

from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController
from wheeledSim.front_camera_sensor import FrontCameraSensor
from wheeledSim.lidar_sensor import LidarSensor


class WheeledSimEnv:
    """
    Wrapper class to make a gym environment from pybullet sim
    """

    def __init__(self, config_file, T=-1, render=True):

        # open file with configuration parameters
        stream = open(config_file, 'r')
        config = yaml.load(stream, Loader=yaml.FullLoader)

        self.client = pybullet.connect(pybullet.GUI) if render else pybullet.connect(pybullet.DIRECT)
        self.robot = Clifford(params=config['cliffordParams'], physicsClientId=self.client)

        # initialize all sensors from config file
        sensors = []
        sense_dict = config['sensors']
        for sensor in config['sensors']:
            # TODO: Test (I can't import rospy so I can't currently test this to see if it works, but this is the idea)
            sensors.append(sense_dict[sensor]['init'](self.robot, senseParamsIn=sense_dict[sensor]['params'],
                                                      physicsClientId=self.client))

        # load simulation environment
        self.env = simController(self.robot, self.client, config['simulationParams'], config['senseParams'],
                                 config['terrainMapParams'], config['terrainParams'], config['explorationParams'])

        self.T = T  # max steps allowed
        self.nsteps = 0  # number of steps taken

    @property
    def observation_space(self):
        # observation takes form (x,y,z) position, (x,y,z,w) quaternion orientation, velocity, joint state
        state_space = gym.spaces.Box(low=np.ones(13) * -float('inf'), high=np.ones(13) + float('inf'))

        # TODO: Incorporate sensing data into state space?
        """
        if not self.use_images:
            return state_space
        else:
            image_space = gym.spaces.Box(low=np.ones(self.env.senseParams['senseResolution']) * -float('inf'),
                                         high=np.ones(self.env.senseParams['senseResolution']) * float('inf'))
            return gym.spaces.Dict({'state': state_space, 'image': image_space})
        """
        return state_space

    @property
    def action_space(self):
        # action takes form [throttle, steering]
        return gym.spaces.Box(low=-np.ones(2), high=np.ones(2))

    def reset(self):
        # reset environment
        self.env.newTerrain()
        self.env.resetRobot()
        self.nsteps = 0

        # get new observation
        obs = self.env.getObservation()

        return np.array(obs)

    def step(self, action):
        # get state action, next state, and boolean terminal state from simulation
        state_action, next_state, sim_t = self.env.controlLoopStep(action)  # image is sa[1]

        # TODO: figure out incorporating sensing data to observation
        obs = {"state": np.array(next_state)[0]}

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
    config_file = "../configurations/cliffordExampleParams.yaml"
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
