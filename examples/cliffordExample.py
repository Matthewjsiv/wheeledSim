import pybullet as p
import numpy as np
from wheeledSim.simController import simController
from wheeledSim.shock_travel_sensor import ShockTravelSensor
from wheeledSim.front_camera_sensor import FrontCameraSensor
from wheeledSim.lidar_sensor import LidarSensor
from wheeledRobots.clifford.cliffordRobot import Clifford

if __name__=="__main__":
    """start pyBullet"""
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    """initialize clifford robot"""
    cliffordParams={"maxThrottle":20, # dynamical parameters of clifford robot
                    "maxSteerAngle":0.5,
                    "susOffset":-0.00,
                    "susLowerLimit":-0.01,
                    "susUpperLimit":0.00,
                    "susDamping":10,
                    "susSpring":500,
                    "traction":1.25,
                    "massScale":1.0}
    robot = Clifford(params=cliffordParams,physicsClientId=physicsClient)

    # sensors = [ShockTravelSensor(robot, physicsClient),FrontCameraSensor(robot, physicsClient),LidarSensor(robot, physicsClient) ]
    sensors = [FrontCameraSensor(robot, physics_client_id=physicsClient),LidarSensor(robot, physics_client_id=physicsClient) ]

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
    terrainMapParams = {"mapWidth":512, # width of matrix
                    "mapHeight":512, # height of matrix
                    "widthScale":0.14, # each pixel corresponds to this distance
                    "heightScale":0.14,
                    "depthScale":28}
    terrainParams = {"AverageAreaPerCell":1.0,
                    "cellPerlinScale":5,
                    "cellHeightScale":0.6, # parameters for generating terrain
                    "smoothing":0.7,
                    "perlinScale":2.5,
                    "perlinHeightScale":0.1,
                    "terrainType": "mountains"}

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

    # import pdb;pdb.set_trace()
    # initialize simulation controller
    sim = simController(robot,simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,explorationParamsIn= explorationParams,physicsClientId=physicsClient, sensors=sensors)
    # save simulation parameters for future reuse (sim params, robot params, terrain map params, terrain params, sensing params)
    #np.save('exampleAllSimParams.npy',[sim.simulationParams,robot.params,sim.terrain.terrainMapParams,sim.terrainParams,sim.senseParams])
    plotSensorReadings = False # plot sensor reading during simulation?
    plotPos = False
    if plotSensorReadings or plotPos:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        if plotPos:
            TERRAIN_IMG = plt.imread('wheeledSim/gimp_overlay_out.png')#.transpose(1,0,2)
            TERRAIN_IMG = plt.imread('wheeledSim/frictionRectangle.png')#.transpose(1,0,2)
            TERRAIN_IMG[:,:] = np.fliplr(TERRAIN_IMG[:,:])
    # simulate trajectory of length 100
    for i in range(10000):
        # step simulation
        action = sim.randomDriveAction()
        # action = [0,0]
        # print(action)

        data = sim.controlLoopStep(action)
        if data[2]: # simulation failed, restartsim
            sim.newTerrain()
            sim.resetRobot()
        else:
            if plotSensorReadings:
                sensorData = data[0][1]
                plt.clf()
                if sim.senseParams["senseType"] == 2: #point cloud
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(sensorData[0,:],sensorData[1,:],sensorData[2,:],s=0.1,c='r',marker='o')
                    ax.scatter(sensorData[0,:],sensorData[1,:],sensorData[2,:],s=.3,c=sensorData[2,:],marker='o',cmap=plt.get_cmap('viridis'))
                    ax.set_xlim([-5,5])
                    ax.set_ylim([-5,5])
                    ax.set_zlim([-5,5])
                else: # 2d map
                    ax = fig.add_subplot()
                    ax.imshow(sensorData,aspect='auto')
                #plt.show()
                plt.draw()
                plt.pause(0.001)
            elif plotPos:
                ax = fig.add_subplot()
                # ax.imshow(TERRAIN_IMG,aspect='auto')
                pose = data[1]
                # print(pose)
                ax.plot(pose[0][0]/.14 + 512/2,pose[0][1]/.14 + 512/2,'r.')
                plt.draw()
                plt.pause(.001)

        # viewMatrix = p.computeViewMatrix(
        #     cameraEyePosition=[0, 0, 3],
        #     cameraTargetPosition=[0, 0, 0],
        #     cameraUpVector=[0, 1, 0])
        # projectionMatrix = p.computeProjectionMatrixFOV(
        #     fov=45.0,
        #     aspect=1.0,
        #     nearVal=0.1,
        #     farVal=3.1)
        # p.getCameraImage(224,224, viewMatrix=viewMatrix,projectionMatrix=projectionMatrix)

    # end simulation
    p.disconnect()
