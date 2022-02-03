import time
import os
import yaml
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

from wheeledSim.simController import simController
from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledRobots.simple_car.simple_car import SimpleCar

def maybe_mkdir(fp, force=True):
    if not os.path.exists(fp):
        os.mkdir(fp)
    elif not force:
        x = input('{} already exists. Hit enter to continue and overwrite. Q to exit.'.format(fp))
        if x.lower() == 'q':
            exit(0)

if __name__=="__main__":
    """start pyBullet"""
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    """initialize clifford robot"""
    cliffordParams={"maxThrottle":100, # dynamical parameters of clifford robot
                    "maxSteerAngle":0.5,
                    "traction":1.25,
                    "massScale":1.0}
#    robot = Clifford(params=cliffordParams,physicsClientId=physicsClient)

    """initialize simulation controls (terrain, robot controls, sensing, etc.)"""
    # physics engine parameters
    simParams = {"timeStep":1./500.,
                "stepsPerControlLoop":50,
                "numSolverIterations":300,
                "gravity":-10,
                "contactBreakingThreshold":0.0001,
                "contactSlop":0.0001,
                "moveThreshold":0.1,
                "maxStopMoveLength":250}
    # random terrain generation parameters
    terrainMapParams = {"mapWidth":500, # width of matrix
                    "mapHeight":500, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1,
                    "depthScale":1}
    terrainParams = {"AverageAreaPerCell":50.0,
                    "cellPerlinScale":1.0,
                    "cellHeightScale":1.0, # parameters for generating terrain
                    "smoothing":2.0,
                    "perlinScale":0.2,
                    "perlinHeightScale":0.0,
                    "frictionScale":0.0,
                    "frictionOffset":1.0
                    }

    explorationParams = {"explorationType":"boundedExplorationNoise"}
    # robot sensor parameters
    heightMapSenseParams = {} # use base params for heightmap
    lidarDepthParams = {"senseDim":[2.*np.pi,np.pi/4.], # angular width and height of lidar sensing
                    "lidarAngleOffset":[0,0], # offset of lidar sensing angle
                    "lidarRange":120, # max distance of lidar sensing
                    "senseResolution":[512,16], # resolution of sensing (width x height)
                    "removeInvalidPointsInPC":False, # remove invalid points in point cloud
                    "senseType":1,
                    "sensorPose":[[0,0,0.3],[0,0,0,1]]} # pose of sensor relative to body
    lidarDepthParams = {"senseDim":[2.*np.pi,np.pi/4.], # angular width and height of lidar sensing
                    "lidarAngleOffset":[0,0], # offset of lidar sensing angle
                    "lidarRange":120, # max distance of lidar sensing
                    "senseResolution":[512,64], # resolution of sensing (width x height)
                    "removeInvalidPointsInPC":False, # remove invalid points in point cloud
                    "senseType":1,
                    "sensorPose":[[0,0,0.3],[0,0,0,1]]} # pose of sensor relative to body
    lidarPCParams = lidarDepthParams.copy()
    lidarPCParams["senseType"] = 2
    noSenseParams = {"senseType":-1}
    senseParams = noSenseParams # use this kind of sensing

    robot = SimpleCar(t_params=terrainMapParams, params=cliffordParams,physicsClientId=physicsClient)
    sensors = []

    cell_height_scales = [0.0, 1.0, 0.5, 0.5, 0.0]
    noise_height_scales = [0.0, 0.1, 0.1, 0.5, 1.0]

    experiment_name = 'simulation_benchmark_data'
    maybe_mkdir('simulation_benchmark_data')
    maybe_mkdir('simulation_benchmark_data/terrains')
    maybe_mkdir('simulation_benchmark_data/actions')

    for i, (ch, nh) in enumerate(zip(cell_height_scales, noise_height_scales)):
        dir_name = 'terrains/type_{}'.format(i+1)
        maybe_mkdir(os.path.join(experiment_name, dir_name))
        terrainParams['cellHeightScale'] = ch
        terrainParams['perlinHeightScale'] = nh

        with open(os.path.join(experiment_name, dir_name, 'terrain_map_params.yaml'), 'w') as fh:
            fh.write(yaml.dump(terrainMapParams))

        with open(os.path.join(experiment_name, dir_name, 'terrain_params.yaml'), 'w') as fh:
            fh.write(yaml.dump(terrainParams))

        sim = simController(robot,simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,explorationParamsIn= explorationParams,physicsClientId=physicsClient, sensors=sensors)

        for j in range(10):
            sim.newTerrain()
            sim.resetRobot()
            heightmap = sim.terrain.gridZ
            np.save(os.path.join(experiment_name, dir_name, 'heightmap_{}'.format(j+1)), heightmap)

    all_actions = []
    for i in range(100):
        actions = []
        for j in range(200):
            action = sim.randomDriveAction()
            actions.append(action)
        actions = np.stack(actions)
        all_actions.append(actions)

    for i, actions in enumerate(all_actions):
        np.save(os.path.join('simulation_benchmark_data/actions', 'sequence_{}'.format(i+1)), actions)

    # end simulation
    p.disconnect()

