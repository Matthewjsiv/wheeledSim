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

    def __init__(self, use_images=False, simulationParamsIn = {}, senseParamsIn={}, terrainMapParamsIn={},
                 terrainParamsIn={}, existingTerrain=None, cliffordParams={}, explorationParamsIn={},T=-1, render=True):

        self.client = pybullet.connect(pybullet.GUI) if render else pybullet.connect(pybullet.DIRECT)
        self.robot = Clifford(params=cliffordParams, physicsClientId=self.client)
        if existingTerrain:
            existingTerrain.generate()
        self.env = simController(self.robot, self.client, simulationParamsIn, senseParamsIn, terrainMapParamsIn,
                                 terrainParamsIn, explorationParamsIn)
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
        timeout = (self.T > 0) and (self.nsteps >= self.T)
        reward = self.reward(obs)
        return obs, reward, sim_t or timeout, {}

    def reward(self, obs):
        # can be updated with a reward function in future work
        return 0


if __name__ == '__main__':

    """load configuration parameters"""
    stream = open("../configurations/cliffordExampleParams.yaml", 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    """initialize clifford robot"""
    cliffordParams = config['cliffordParams']

    """initialize simulation controls (terrain, robot controls, sensing, etc.)"""
    # physics engine parameters
    simParams = config['simParams']

    print(simParams['timeStep'])

    # random terrain generation parameters
    terrainMapParams = config['terrainMapParams']
    terrainParams = config['terrainParams']

    explorationParams = {"explorationType": "boundedExplorationNoise"}

    # robot sensor parameters
    heightMapSenseParams = {}  # use base params for heightmap
    lidarDepthParams = config['lidarDepthParams']
    lidarPCParams = lidarDepthParams.copy()
    lidarPCParams["senseType"] = 2
    noSenseParams = {"senseType": -1}
    senseParams = noSenseParams  # use this kind of sensing

    env = WheeledSimEnv(simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,
                        terrainParamsIn=terrainParams,explorationParamsIn=explorationParams, T=50)

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
