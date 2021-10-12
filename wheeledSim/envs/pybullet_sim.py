import gym
import torch
import numpy as np
import pybullet
import yaml

from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController
from wheeledSim.sensors.sensor_map import sensor_str_to_obj

class WheeledSimEnv:
    """
    Wrapper class to make a gym environment from pybullet sim
    """

    def __init__(self, config_file, T=-1, render=True):

        # open file with configuration parameters
        stream = open(config_file, 'r')
        config = yaml.load(stream, Loader=yaml.FullLoader)

        self.client = pybullet.connect(pybullet.GUI, options=config.get('backgroundColor', "")) if render else pybullet.connect(pybullet.DIRECT)
        self.robot = Clifford(params=config['cliffordParams'], t_params = config['terrainMapParams'], physicsClientId=self.client)

        # load simulation environment
        self.env = simController(self.robot, self.client, config['simulationParams'], config['senseParams'],
                                 config['terrainMapParams'], config['terrainParams'], config['explorationParams'])

        # initialize all sensors from config file
        sensors = []
        self.sense_dict = config.get('sensors', {})
        if self.sense_dict is None:
            self.sense_dict = {}

        for sensor in self.sense_dict.values():
            assert sensor['type'] in sensor_str_to_obj.keys(), "{} not a valid sensor type. Valid sensor types are {}".format(sensor['type']. sensor_str_to_obj.keys())

            sensor_cls = sensor_str_to_obj[sensor['type']]
            sensor = sensor_cls(self.env, senseParamsIn=sensor.get('params', {}), topic=sensor.get('topic', None))
            sensors.append(sensor)

        self.env.set_sensors(sensors)

        self.T = T  # max steps allowed
        self.nsteps = 0  # number of steps taken

    @property
    def observation_space(self):
        # observation takes form (x,y,z) position, (x,y,z,w) quaternion orientation, velocity, joint state
        state_dim = 7
        if self.env.senseParams['recordVelocity']:
            state_dim += 6

        state_space = gym.spaces.Box(low=np.ones(state_dim) * -float('inf'), high=np.ones(state_dim) * float('inf'))

        # Add sensor output to observation space dict
        sensor_space = {'state': state_space}
        for sensor in self.env.sensors:
            sensedim = [self.env.stepsPerControlLoop] + sensor.N if sensor.is_time_series else sensor.N
            out_space = gym.spaces.Box(low=np.ones(sensedim)*-float('inf'),
                                       high=np.ones(sensedim)*float('inf'))
            sensor_space[sensor.topic] = out_space

        obs_space = gym.spaces.Dict(sensor_space)

        return obs_space

    @property
    def action_space(self):
        # action takes form [throttle, steering]
        return gym.spaces.Box(low=-np.ones(2), high=np.ones(2))

    def reset(self):
        # reset environment
        self.env.newTerrain()
        self.env.resetRobot()
        self.env.robot.params['frictionMap'] = self.env.terrain.frictionMap
        self.nsteps = 0

        # get new observation
        obs, sensedata = self.env.getObservation()
        sensedata['state'] = torch.tensor(obs).float()

        return sensedata

    def step(self, action):
        # get state action, next state, and boolean terminal state from simulation
        state_action, next_state, sim_t = self.env.controlLoopStep(action)

        # TODO: clean up after checking in about changing control loop function
        obs = next_state[1]  # sensing data
        obs["state"] = torch.tensor(next_state[0][0]).float()

        # increment number of steps and set terminal state if reached max steps
        self.nsteps += 1
        timeout = self.get_terminal()
        reward = self.get_reward(obs)

        return obs, reward, sim_t or timeout, {}

    def get_reward(self, obs):
        # can be updated with a reward function in future work
        return torch.tensor(0.)

    def get_terminal(self):
        return (self.T > 0) and (self.nsteps >= self.T)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """load environment"""
    config_file = "../configurations/cliffordExampleParams.yaml"
    env = WheeledSimEnv(config_file, T=50, render=True)

    test = env.observation_space
    print('Observation:', test)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plt.show(block=False)

    # run simulation 5 times
    for _ in range(5):
        terminal = False
        t = 0
        while not terminal:
            t += 1
#            a = env.action_space.sample()
            a = [1.0, 0.3]
            obs, reward, terminal, i = env.step(a)
            # print('STATE = {}, ACTION = {}, t = {}'.format(obs, a, t))

            for ax in axs.flatten():
                ax.cla()
            axs[0, 0].set_title('Lidar')
            axs[0, 1].set_title('Heightmap')
            axs[1, 0].set_title('Front Camera')
            axs[1, 1].set_title('Shocks')

            axs[0, 0].scatter(obs['lidar'][:, 0], obs['lidar'][:, 1], s=1.,c=obs['lidar'][:,2],cmap=plt.get_cmap('viridis'))
            axs[0, 1].imshow(obs['heightmap'][0,:,:])
            fc = obs['front_camera']
            axs[1, 0].imshow(np.transpose(fc[:, :, :],(1,2,0)))
            for i, l in zip(range(4), ['fl', 'fr', 'bl', 'br']):
                axs[1, 1].plot(obs['shock_travel'][:, i], label='{}_travel'.format(l))
            axs[1, 1].legend()
            plt.pause(1e-2)

        env.reset()
