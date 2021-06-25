import pybullet as p
import numpy as np
from wheeledSim.simController import simController
from wheeledRobots.clifford.cliffordRobot import Clifford
import keyboard

# ROBOT = []
STEER = 0
THROTTLE = 0
def handleLeftKey(e):
    global STEER
    STEER -= .08
    STEER = max(-1,STEER)
def handleRightKey(e):
    global STEER
    STEER += .08
    STEER = min(1,STEER)
def handleTurnRelease(e):
    ...
    # work your magic
def handleUpKey(e):
    global THROTTLE
    THROTTLE += .03
    THROTTLE = min(2,THROTTLE)
def handleThrottleRelease(e):
    global THROTTLE
    THROTTLE -= .03
    THROTTLE = max(0,THROTTLE)

if __name__=="__main__":
    """start pyBullet"""
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    fm = np.ones([512,512]) + 1
    # fm[:int(512/2),:] = 0
    #FRICTION MAP IN SAME FRAME AS IMAGE, so needs to be flipped as well
    # fm = np.load('wheeledSim/frictionCheckerboard.npy')
    # fm = np.load('wheeledSim/frictionRectangle.npy')
    # fm = np.fliplr(fm)
    """initialize clifford robot"""
    cliffordParams={"maxThrottle":10, # dynamical parameters of clifford robot
                    "maxSteerAngle":0.5,
                    "susOffset":-0.00,
                    "susLowerLimit":-0.01,
                    "susUpperLimit":0.00,
                    "susDamping":10,
                    "susSpring":500,
                    "traction":1.25,
                    "massScale":1.0,
                    "frictionMap": fm}

    keyboard.on_press_key("left", handleLeftKey)
    keyboard.on_release_key("left", handleTurnRelease)
    keyboard.on_press_key("right", handleRightKey)
    keyboard.on_release_key("right", handleTurnRelease)
    keyboard.on_press_key("up", handleUpKey)
    keyboard.on_release_key("up", handleThrottleRelease)
    keyboard.on_press_key("down", handleThrottleRelease)

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
    terrainMapParams = {"mapWidth":300, # width of matrix
                    "mapHeight":300, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1,
                    "depthScale":1}
    terrainParams = {"AverageAreaPerCell":1.0,
                    "cellPerlinScale":5,
                    "cellHeightScale":0.6, # parameters for generating terrain
                    "smoothing":0.7,
                    "perlinScale":2.5,
                    "perlinHeightScale":0.1,
                    "terrainType": "basicFriction"}
    # robot sensor parameters
    heightMapSenseParams = {} # use base params for heightmap
    lidarDepthParams = {"senseDim":[2.*np.pi,np.pi/4.], # angular width and height of lidar sensing
                    "lidarAngleOffset":[0,0], # offset of lidar sensing angle
                    "lidarRange":120, # max distance of lidar sensing
                    "senseResolution":[512,16], # resolution of sensing (width x height)
                    "removeInvalidPointsInPC":False, # remove invalid points in point cloud
                    "senseType":1,
                    "sensorPose":[[0,0,0.3],[0,0,0,1]]} # pose of sensor relative to body

    lidarPCParams = lidarDepthParams.copy()
    lidarPCParams["senseType"] = 2
    noSenseParams = {"senseType":-1}
    senseParams = noSenseParams # use this kind of sensing
    # senseParams = lidarPCParams

    robot = Clifford(params=cliffordParams,t_params = terrainMapParams, physicsClientId=physicsClient)

    # initialize simulation controller
    sim = simController(robot,simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,physicsClientId=physicsClient)
    # save simulation parameters for future reuse (sim params, robot params, terrain map params, terrain params, sensing params)
    #np.save('exampleAllSimParams.npy',[sim.simulationParams,robot.params,sim.terrain.terrainMapParams,sim.terrainParams,sim.senseParams])
    plotSensorReadings = False # plot sensor reading during simulation?
    plotPos = False
    if plotSensorReadings or plotPos:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        if plotPos:
            TERRAIN_IMG = plt.imread('wheeledSim/gimp_overlay_out.png')#.transpose(1,0,2)
            #TERRAIN_IMG = plt.imread('wheeledSim/frictionCheckerboard.png')#.transpose(1,0,2)
            TERRAIN_IMG[:,:] = np.fliplr(TERRAIN_IMG[:,:])
    # simulate trajectory of length 100
    for i in range(10000):
        # step simulation
        action = [THROTTLE, STEER]
        # print(robot.getTirePos())
        # print(action)
        data = sim.controlLoopStep(action)
        if data[2]: # simulation failed, restartsim
            sim.newTerrain()
            sim.resetRobot()
        else:
            if plotSensorReadings:
                sensorData = data[0][1]
                # print(sensorData.shape)
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
                plt.clf()
                ax = fig.add_subplot()
                ax.imshow(TERRAIN_IMG)
                pose = data[1]
                # print(pose)
                # ax.plot(pose[0][0]/.14 + 512/2,pose[0][1]/.14 + 512/2,'r.',markersize=.1)
                # ax.imshow(TERRAIN_IMG[int(pose[0][0]/.14 + 512/2) - 20:int(pose[0][0]/.14 + 512/2) + 20,int(pose[0][1]/.14 + 512/2) - 20:int(pose[0][1]/.14 + 512/2) + 20])
                poses = robot.getTirePos()
                # print(poses[0])
                for t in range(4):
                    ax.plot(poses[t][0][0]/.14 + 512/2,poses[t][0][1]/.14 + 512/2,'r.',markersize=1)
                    # ax.plot(poses[t][0][1]/.14 + 512/2,poses[t][0][0]/.14 + 512/2,'r.',markersize=1)
                    # print(t)
                ax.invert_yaxis()
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
