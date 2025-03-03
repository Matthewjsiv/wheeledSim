import time
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

from wheeledSim.simController import simController
from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor
from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
from wheeledSim.sensors.lidar_sensor import LidarSensor
from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledRobots.simple_car.simple_car import SimpleCar

if __name__=="__main__":
    """start pyBullet"""
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    """initialize clifford robot"""
    cliffordParams={"maxThrottle":20, # dynamical parameters of clifford robot
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
    terrainMapParams = {"mapWidth":100, # width of matrix
                    "mapHeight":100, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1,
                    "depthScale":1}
    terrainParams = {"AverageAreaPerCell":10.0,
                    "cellPerlinScale":1.0,
                    "cellHeightScale":1.0, # parameters for generating terrain
                    "smoothing":0.7,
                    "perlinScale":0.5,
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
    # senseParams = lidarPCParams

    robot = SimpleCar(t_params=terrainMapParams, params=cliffordParams,physicsClientId=physicsClient)
#    robot = Clifford(t_params=terrainMapParams, params=cliffordParams,physicsClientId=physicsClient)

    # sensors = [ShockTravelSensor(robot, physicsClient),FrontCameraSensor(robot, physicsClient),LidarSensor(robot, physicsClient) ]
#    sensors = [FrontCameraSensor(robot, physics_client_id=physicsClient),LidarSensor(robot, physics_client_id=physicsClient) ]
    sensors = []

    # initialize simulation controller
    sim = simController(robot,simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,explorationParamsIn= explorationParams,physicsClientId=physicsClient, sensors=sensors)

    for i in range(10):
        sim.newTerrain()
        sim.resetRobot()
        heightmap = sim.terrain.gridZ
        plt.imshow(heightmap)
        plt.show()

    controls = []
    x = 0.
    cnt = 0
    while x < 50.:
        action = sim.randomDriveAction()
        obs = sim.controlLoopStep([1.0, 0.0])
        x = obs[0][0][0]
        print(x)
        cnt += 1
        time.sleep(0.1)

    print(cnt)

    # end simulation
    p.disconnect()

