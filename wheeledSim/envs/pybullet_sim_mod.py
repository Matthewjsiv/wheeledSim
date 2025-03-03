import gym
import numpy as np
import pybullet
import yaml
import time

from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController
from wheeledSim.front_camera_sensor import FrontCameraSensor
from wheeledSim.lidar_sensor import LidarSensor
from wheeledSim.local_heightmap_sensor import LocalHeightmapSensor
from wheeledSim.shock_travel_sensor import ShockTravelSensor

sensor_str_to_obj = {
    'FrontCameraSensor':FrontCameraSensor,
    'LidarSensor':LidarSensor,
    'LocalHeightmapSensor':LocalHeightmapSensor,
    'ShockTravelSensor':ShockTravelSensor
}

"""
Slight modifications for getting step response
"""

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

        # load simulation environment
        self.env = simController(self.robot, self.client, config['simulationParams'], config['senseParams'],
                                 config['terrainMapParams'], config['terrainParams'], config['explorationParams'])

        # initialize all sensors from config file
        sensors = []
        self.sense_dict = config['sensors']

        for sensor in config['sensors'].values():
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
        state_space = gym.spaces.Box(low=np.ones(13) * -float('inf'), high=np.ones(13) * float('inf'))

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
        self.nsteps = 0

        # get new observation
        obs = self.env.getObservation()

        return np.array(obs)

    def step(self, action):
        # get state action, next state, and boolean terminal state from simulation
        state_action, next_state, sim_t = self.env.controlLoopStep(action)
        # print(next_state)
        # TODO: clean up after checking in about changing control loop function
        obs = next_state[1]  # sensing data
        obs["state"] = np.array(next_state)[0]
        # print(len(obs["state"]))
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
    import matplotlib.pyplot as plt

    """load environment"""
    config_file = "../configurations/cliffordExampleParams.yaml"
    env = WheeledSimEnv(config_file, T=2200, render=True)

    test = env.observation_space
    print('Observation:', test)

    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # plt.show(block=False)

    # run simulation 5 times
    svals = np.array([])
    avals = np.array([])
    tvals = np.array([])
    now = time.perf_counter()
    start = now
    for _ in range(1):
        terminal = False
        t = 0
        while not terminal:
            t += 1
#            a = env.action_space.sample()
            a = [1.0, 0.0]
            obs, reward, terminal, i = env.step(a)
            # print('STATE = {}, ACTION = {}, t = {}'.format(obs, a, t))

            # for ax in axs.flatten():
            #     ax.cla()
            # axs[0, 0].set_title('Lidar')
            # axs[0, 1].set_title('Heightmap')
            # axs[1, 0].set_title('Front Camera')
            # axs[1, 1].set_title('Shocks')
            #
            # axs[0, 0].scatter(obs['lidar'][:, 0], obs['lidar'][:, 1], s=1.)
            # axs[0, 1].imshow(obs['heightmap'])
            # axs[1, 0].imshow(obs['front_camera'][:, :, :3])
            # for i, l in zip(range(4), ['fl', 'fr', 'bl', 'br']):
            #     axs[1, 1].plot(obs['shock_travel'][:, i], label='{}_travel'.format(l))
            # print(obs['state'][7])
            # axs[1, 1].legend()
            # plt.pause(1e-2)
            speed = np.linalg.norm(obs['state'][7:10])
            # print(speed)
            svals = np.hstack([svals,speed])
            avals = np.hstack([avals,40.0/10.0])
            # print(t)
            time_diff = time.perf_counter()- now
            tvals = np.hstack([tvals,time_diff])
            # print(time.perf_counter() - start)
            now = time.perf_counter()
            if t >= 1000:
                print('here')
                np.save('../sysid_data/s_40',svals)
                np.save('../sysid_data/a_40',avals)
                np.save('../sysid_data/t_40',tvals)
                break

        env.reset()
