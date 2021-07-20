import pybullet as p
import numpy as np
import torch
from wheeledSim.terrain.randomTerrain import *
from wheeledSim.randomExplorationPolicy import *

class simController:
    # this class controls the simulation. It controls the terrain and robot, and returns data
    def __init__(self,robot,physicsClientId=0,simulationParamsIn={},senseParamsIn={},
                terrainMapParamsIn={},terrainParamsIn={},explorationParamsIn={},sensors=[]):
        # set up simulation params
        self.simulationParams = {"timeStep":1./500.,
                            "stepsPerControlLoop":50,
                            "numSolverIterations":300,
                            "gravity":-10,
                            "contactBreakingThreshold":0.0001,
                            "contactSlop":0.0001,
                            "moveThreshold":0,
                            "maxStopMoveLength":np.inf,
                            "terminateIfFlipped":False,
                            "randomActionScale":[1,1]}
        self.simulationParams.update(simulationParamsIn)

        # set up robot sensing parameters
        self.senseParams = {"senseDim":[5,5], # width (meter or angle) and height (meter or angle) of terrain map or point cloud
                            "lidarAngleOffset":[0,0],
                            "lidarRange":10,
                            "senseResolution":[100,100], # array giving resolution of map output (num pixels wide x num pixels high)
                            "removeInvalidPointsInPC":False, # remove invalid points in point cloud
                            "senseType":-1, # 0 for terrainMap, 1 for lidar depth image, 2 for lidar point cloud, -1 for nothing
                            "sensorPose":[[0,0,0],[0,0,0,1]], # pose of sensor relative to body
                            "recordJointStates":False} # whether to record joint data or not
        self.senseParams.update(senseParamsIn)

        # set up simulation
        self.physicsClientId=physicsClientId
        self.timeStep = self.simulationParams["timeStep"]
        # Each control loop makes this many simulation steps. The period of a control loop is timeStep*stepsPerControlLoop
        self.stepsPerControlLoop=self.simulationParams["stepsPerControlLoop"]
        p.setPhysicsEngineParameter(numSolverIterations=self.simulationParams["numSolverIterations"],
            contactBreakingThreshold=self.simulationParams["contactBreakingThreshold"],contactSlop=self.simulationParams["contactSlop"],
            physicsClientId=self.physicsClientId)
        p.setGravity(0,0,self.simulationParams["gravity"],physicsClientId=self.physicsClientId)
        p.setTimeStep(self.timeStep,physicsClientId=self.physicsClientId)

        self.sensors = sensors

        # set up terrain
        self.terrainParamsIn = {"terrainType": "randomRockyTerrain",
                            "existingTerrain": None}
        self.terrainParamsIn.update(terrainParamsIn)
        if self.terrainParamsIn["existingTerrain"]!=None:
            self.terrain = self.terrainParamsIn["existingTerrain"]
        else:
            print(self.terrainParamsIn["terrainType"])
            if self.terrainParamsIn["terrainType"] == "randomRockyTerrain":
                self.terrain = randomRockyTerrain(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            elif self.terrainParamsIn["terrainType"] == "randomSloped":
                self.terrain = randomSloped(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            elif self.terrainParamsIn["terrainType"] == "fixSloped":
                self.terrain = fixSloped(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            elif self.terrainParamsIn["terrainType"] == "mountains":
                self.terrain = Mountains(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            elif self.terrainParamsIn["terrainType"] == "flatLand":
                self.terrain = Flatland(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            elif self.terrainParamsIn["terrainType"] == "basicFriction":
                self.terrain = basicFriction(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            elif self.terrainParamsIn["terrainType"] == "obstacles":
                    self.terrain = obstacleCourse(terrainMapParamsIn,physicsClientId=self.physicsClientId)
            else:
                self.terrain = self.terrainParamsIn["terrainType"](terrainMapParamsIn,physicsClientId=self.physicsClientId)
            self.newTerrain()

        # set up determination of wheter robot is stuck
        self.moveThreshold = self.simulationParams["moveThreshold"]*self.simulationParams["moveThreshold"] # store square distance for easier computation later
        self.lastX = 0
        self.lastY = 0
        self.stopMoveCount =0

        # set up random driving
        explorationParams = {"explorationType":"fixedRandomAction"}
        explorationParams.update(explorationParamsIn)
        if explorationParams["explorationType"] == "boundedExplorationNoise":
            self.randDrive = boundedExplorationNoise(explorationParams)
        elif explorationParams["explorationType"] == "fixedRandomAction":
            self.randDrive = fixedRandomAction(explorationParams)
        #self.randDrive = ouNoise()
        #self.randDrive = np.zeros(2)

        # set up robot
        self.camFollowBot = False
        self.robot = robot
        self.lastStateRecordFlag = False # Flag to tell if last state of robot has been recorded or not
        if self.terrainParamsIn["terrainType"] == "randomRockyTerrain" or self.terrainParamsIn["terrainType"] == "basicFriction":
            self.robot.params["frictionMap"] = self.terrain.frictionMap
        self.resetRobot()

    def set_sensors(self, sensors):
        self.sensors = sensors
        self.buf = {k:[torch.zeros(*k.N)] * self.stepsPerControlLoop for k in self.sensors if k.is_time_series}

    # generate new terrain
    def newTerrain(self,**kwargs):
        self.terrain.generate(self.terrainParamsIn,**kwargs)

    # reset the robot
    def resetRobot(self,doFall=True,pos=[0,0],orien=[0,0,0,1]):
        self.controlLoopStep([0,0])
        if len(pos)>2:
            safeFallHeight = pos[2]
        else:
            safeFallHeight = self.terrain.maxLocalHeight(pos,1)+1.3
        self.robot.reset([[pos[0],pos[1],safeFallHeight],orien])
        if doFall:
            fallTime=0.5
            fallSteps = int(np.ceil(fallTime/self.timeStep))
            for i in range(fallSteps):
                self.stepSim()
        self.stopMoveCount = 0
        self.randDrive.reset()

    def stepSim(self):
        self.robot.updateSpringForce()
        self.robot.updateTraction4Tire()
        p.stepSimulation(physicsClientId=self.physicsClientId)
        self.lastStateRecordFlag = False
        if self.camFollowBot:
            pose = self.robot.getPositionOrientation()
            pos = pose[0]
            orien = pose[1]
            forwardDir = p.multiplyTransforms([0,0,0],orien,[1,0,0],[0,0,0,1])[0]
            headingAngle = np.arctan2(forwardDir[1],forwardDir[0])*180/np.pi-90
            p.resetDebugVisualizerCamera(1.0,headingAngle,-15,pos,physicsClientId=self.physicsClientId)

    def controlLoopStep(self,driveCommand):
        # import pdb;pdb.set_trace()
        throttle = driveCommand[0]
        steering = driveCommand[1]
        # Create Prediction Input
        # check if last pose of robot has been recorded
        if not self.lastStateRecordFlag:
            self.lastPose = self.robot.getPositionOrientation()
            self.lastVel = self.robot.getBaseVelocity_body()
            if self.senseParams["recordJointStates"]:
                self.lastJointState = self.robot.measureJoints()
                self.lastAbsoluteState = list(self.lastPose[0])+list(self.lastPose[1])+self.lastVel[:] + self.lastJointState[:]
            else:
                self.lastAbsoluteState = list(self.lastPose[0])+list(self.lastPose[1])+self.lastVel[:]
        #simulate sensing (generate height map or lidar point cloud)
        sensingData = self.sensing(self.lastPose)
        # store state-action for motion prediction
        stateActionData = [self.lastAbsoluteState,sensingData,driveCommand] #(absolute robot state, sensing data, action)
        # command robot throttle & steering and simulate
        self.robot.drive(throttle)
        self.robot.steer(steering)

        data = {k:[] for k in self.sensors}

        for i in range(self.stepsPerControlLoop):
            self.stepSim()
            for sensor in self.sensors:
                if sensor.is_time_series:
                    data[sensor].append(sensor.measure())

        sense_data = {k.topic:torch.stack(v, dim=0) if k.is_time_series else k.measure() for k,v in data.items()}

        # Record outcome state
        newPose = self.robot.getPositionOrientation()
        # check how long robot has been stuck
        if (newPose[0][0]-self.lastX)*(newPose[0][0]-self.lastX) + (newPose[0][1]-self.lastY)*(newPose[0][1]-self.lastY)> self.moveThreshold:
            self.lastX = newPose[0][0]
            self.lastY = newPose[0][1]
            self.stopMoveCount = 0
        else:
            self.stopMoveCount +=1

        self.lastAbsoluteState = self.getObservation()
        #self.lastStateRecordFlag = True
        newStateData = [self.lastAbsoluteState,sense_data]
        return stateActionData,newStateData,self.simTerminateCheck(newPose)

    def getObservation(self):
        # relative position, body twist, joint position and velocity
        pose = self.robot.getPositionOrientation()
        vel = self.robot.getBaseVelocity_body()
        joints = self.robot.measureJoints()
        if self.senseParams['recordJointStates']:
            obs = list(pose[0]) + list(pose[1]) + vel[:] + joints[:]
        else:
            obs = list(pose[0]) + list(pose[1]) + vel[:]

        sense_data = {s.topic:torch.stack([s.measure() for _ in range(self.stepsPerControlLoop)], dim=0) if s.is_time_series else s.measure() for s in self.sensors}

        return [obs, sense_data]

    # check if simulation should be terminated
    def simTerminateCheck(self,robotPose):
        termSim = False
        # flipped robot termination criteria
        if self.simulationParams["terminateIfFlipped"]:
            upDir = p.multiplyTransforms([0,0,0],robotPose[1],[0,0,1],[0,0,0,1])[0]
            if upDir[2] < 0:
                termSim = True
        # stuck robot terminate criteria
        if self.stopMoveCount > self.simulationParams["maxStopMoveLength"]:
            termSim = True
        # boundary criteria
        minZ = np.min(self.terrain.gridZ) - 100.
        maxZ = np.max(self.terrain.gridZ) + 100.
        minX = np.min(self.terrain.gridX) + 1.
        maxX = np.max(self.terrain.gridX) - 1.
        minY = np.min(self.terrain.gridY) + 1.
        maxY = np.max(self.terrain.gridY) - 1.
        if robotPose[0][0] > maxX or robotPose[0][0] < minX or \
        robotPose[0][1] > maxY or robotPose[0][1] < minY or \
        robotPose[0][2] > maxZ or robotPose[0][2] < minZ:
            termSim = True
        return termSim

    # generate sensing data
    def sensing(self,robotPose,senseType=None,expandDim=False):
        # # viewMatrix = p.computeViewMatrix(
        # #     cameraEyePosition=robotPose,
        # #     cameraTargetPosition=[0, 0, 0],
        # #     cameraUpVector=[0, 1, 0])
        # projectionMatrix = p.computeProjectionMatrixFOV(
        #     fov=45.0,
        #     aspect=1.0,
        #     nearVal=0.1,
        #     farVal=3.1)
        # # p.getCameraImage(224,224, viewMatrix=viewMatrix,projectionMatrix=projectionMatrix)
        # rotation = p.getMatrixFromQuaternion(robotPose[1])
        # forward_vector = [rotation[0], rotation[3], rotation[6]]
        # up_vector = [rotation[2], rotation[5], rotation[8]]
        #
        # camera_target = [
        #     robotPose[0][0] + forward_vector[0] * 10,
        #     robotPose[0][1] + forward_vector[1] * 10,
        #     robotPose[0][2] + forward_vector[2] * 10]
        #
        # #forward shift (hopefully?)
        # m = 1
        # frontPose = [robotPose[0][0] + m*(2*robotPose[1][1]*robotPose[1][3] - 2*robotPose[1][2]*robotPose[1][0]),
        #             robotPose[0][1] + m*(2*robotPose[1][2]*robotPose[1][3] + 2*robotPose[1][1]*robotPose[1][0]),
        #             robotPose[0][2] + m*(1 - 2*robotPose[1][1]**2 - 2*robotPose[1][2]**2)]
        #
        # # view_matrix = p.computeViewMatrix(
        # #     [robotPose[0][0],robotPose[0][1],robotPose[0][2]],
        # #     camera_target,
        # #     up_vector,
        # #     physicsClientId=self.physicsClientId)
        # view_matrix = p.computeViewMatrix(
        #     frontPose,
        #     camera_target,
        #     up_vector,
        #     physicsClientId=self.physicsClientId)

        ##Or using camfollowbot setup
        # pose = self.robot.getPositionOrientation()
        # posx,posy,posz = pose[0]
        # orien = pose[1]
        # forwardDir = p.multiplyTransforms([0,0,0],orien,[1,0,0],[0,0,0,1])[0]
        # pos[0] = pos[0] + forwardDir[0]
        # pos[1] = pos[1] + forwardDir[1]
        # pos[2] = pos[2] + forwardDir[2]


        # pose = self.robot.getPositionOrientation()
        # posx,posy,posz = pose[0][0],pose[0][1],pose[0][2]
        posx,posy,posz = robotPose[0][0],robotPose[0][1],robotPose[0][2]
        # orien = pose[1]
        # forwardDir = p.multiplyTransforms([0,0,0],orien,[1,0,0],[0,0,0,1])[0]

        rotation = p.getMatrixFromQuaternion(robotPose[1])
        forwardDir = [rotation[0], rotation[3], rotation[6]]
        upDir = [rotation[2], rotation[5], rotation[8]]

        # m = 1
        # posx += forwardDir[0]*(m + .05)
        # posy += forwardDir[1]*(m+.05)
        # posz += forwardDir[2]*(m + .08)
        posx += forwardDir[0]*.12
        posy += forwardDir[1]*.12
        posz += forwardDir[2]*.12
        posx += upDir[0]*.12
        posy += upDir[1]*.12
        posz += upDir[2]*.12
        # posz += 3
        # headingAngle = np.arctan2(forwardDir[1],forwardDir[0])*180/np.pi-90
        # pitchAngle = forwardDir[2]*180/np.pi
        q = robotPose[1]
        # rollAngle = -1*np.arcsin(2*q[0]*q[1] + 2*q[2]*q[3])*180/np.pi
        # rollAngle = np.arctan2(2*quat[1]*quat[3] - 2*quat[0]*quat[2], 1- 2*quat[1]*quat[1] - 2*quat[2]*quat[2])*180/np.pi
        #roll = atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z);
        #wxyz or xyzw?
        #quat won't work bc global
        # rollAngle = np.arctan2(2.0*(q[0]*q[1] + q[3]*q[2]),q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2])*180/np.pi
        # pitchAngle = -1*(np.arcsin(forwardDir[2])*180/np.pi - 90)
        # print(rollAngle)
        # rollAngle = 180
        # pitchAngle = 0
        rollAngle = np.arctan2(2.0 * (q[2]*q[1] + q[3]*q[0]),1.0 - 2.0*(q[0]*q[0] + q[1]*q[1]))*180/np.pi
        pitchAngle = -1*np.arcsin(2.0 * (q[1]*q[3] - q[2]*q[0]))*180/np.pi
        headingAngle = np.arctan2(2.0 * (q[2]*q[3] + q[0]*q[1]), -1.0 + 2.0*(q[3]*q[3] + q[0]*q[0]))*180/np.pi - 90
        # rollAngle = 0
        # pitchAngle = 90
        # headingAngle = 0
        # pitchAngle = 90
        view_matrix = p.computeViewMatrixFromYawPitchRoll((posx,posy,posz),.11,headingAngle,pitchAngle,rollAngle,2,physicsClientId=self.physicsClientId)
        projectionMatrix = p.computeProjectionMatrixFOV(fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=18.1)
        # p.resetDebugVisualizerCamera(1.0,headingAngle,-15,pos,physicsClientId=self.physicsClientId)

        w,h,rgbImg,depthImg,segImg = p.getCameraImage(
                400,
                400,
                view_matrix,
                projectionMatrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_NO_SEGMENTATION_MASK,
                physicsClientId=self.physicsClientId)


        if senseType is None:
            senseType = self.senseParams["senseType"]
        if not isinstance(senseType,int):
            return [self.sensing(robotPose,senseType[i],expandDim) for i in range(len(senseType))]
        sensorAbsolutePose = p.multiplyTransforms(robotPose[0],robotPose[1],self.senseParams["sensorPose"][0],self.senseParams["sensorPose"][1])
        if senseType == -1: # no sensing
            sensorData = np.array([])
        elif senseType == 0: #get terrain height map
            sensorData = self.terrain.sensedHeightMap(sensorAbsolutePose,self.senseParams["senseDim"],self.senseParams["senseResolution"])
        else: # get lidar data
            horzAngles = np.linspace(-self.senseParams["senseDim"][0]/2.,self.senseParams["senseDim"][0]/2.,self.senseParams["senseResolution"][0]+1)+self.senseParams["lidarAngleOffset"][0]
            horzAngles = horzAngles[0:-1]
            vertAngles = np.linspace(-self.senseParams["senseDim"][1]/2.,self.senseParams["senseDim"][1]/2.,self.senseParams["senseResolution"][1])+self.senseParams["lidarAngleOffset"][1]
            horzAngles,vertAngles = np.meshgrid(horzAngles,vertAngles)
            originalShape = horzAngles.shape
            horzAngles = horzAngles.reshape(-1)
            vertAngles = vertAngles.reshape(-1)
            sensorRayX = np.cos(horzAngles)*np.cos(vertAngles)*self.senseParams["lidarRange"]
            sensorRayY = np.sin(horzAngles)*np.cos(vertAngles)*self.senseParams["lidarRange"]
            sensorRayZ = np.sin(vertAngles)*self.senseParams["lidarRange"]
            xVec = np.array(p.multiplyTransforms([0,0,0],sensorAbsolutePose[1],[1,0,0],[0,0,0,1])[0])
            yVec = np.array(p.multiplyTransforms([0,0,0],sensorAbsolutePose[1],[0,1,0],[0,0,0,1])[0])
            zVec = np.array(p.multiplyTransforms([0,0,0],sensorAbsolutePose[1],[0,0,1],[0,0,0,1])[0])
            endX = sensorAbsolutePose[0][0]+sensorRayX*xVec[0]+sensorRayY*yVec[0]+sensorRayZ*zVec[0]
            endY = sensorAbsolutePose[0][1]+sensorRayX*xVec[1]+sensorRayY*yVec[1]+sensorRayZ*zVec[1]
            endZ = sensorAbsolutePose[0][2]+sensorRayX*xVec[2]+sensorRayY*yVec[2]+sensorRayZ*zVec[2]
            rayToPositions = np.stack([endX,endY,endZ],axis=0).transpose().tolist()
            rayFromPositions = np.repeat(np.matrix(sensorAbsolutePose[0]),len(rayToPositions),axis=0).tolist()
            rayResults = ()
            while len(rayResults)<len(rayToPositions):
                batchStartIndex = len(rayResults)
                batchEndIndex = batchStartIndex + p.MAX_RAY_INTERSECTION_BATCH_SIZE
                rayResults = rayResults + p.rayTestBatch(rayFromPositions[batchStartIndex:batchEndIndex],rayToPositions[batchStartIndex:batchEndIndex],physicsClientId=self.physicsClientId)
            rangeData = np.array([rayResults[i][2] for i in range(len(rayResults))]).reshape(originalShape)
            if senseType == 1: # return depth map
                sensorData = rangeData
            else: # return point cloud
                lidarPoints = [rayResults[i][3] for i in range(len(rayResults))]
                lidarPoints = np.array(lidarPoints).transpose()
                if self.senseParams["removeInvalidPointsInPC"]:
                    lidarPoints = lidarPoints[:,rangeData.reshape(-1)<1]
                sensorData = lidarPoints


                b2local = p.invertTransform(sensorAbsolutePose[0],sensorAbsolutePose[1])
                # for i in range(sensorData.shape[1]):
                #     test = np.array(p.multiplyTransforms(b2local[0],b2local[1],sensorData[:,i],sensorAbsolutePose[1])[0])
                #     sensorData[:,i] = test

                #vectorized form gives >200 times speed up over for loop using pybullet method
                # :)
                bt = np.expand_dims(np.array(b2local[0]),axis=0)
                bb = np.array(p.getMatrixFromQuaternion(b2local[1])).reshape([3,3])
                sB = np.array(p.getMatrixFromQuaternion(sensorAbsolutePose[1])).reshape([3,3])
                sensorData = bt.T + bb @ sensorData

        if expandDim:
            sensorData = np.expand_dims(sensorData,axis=0)
        return sensorData
    # generate random drive action
    def randomDriveAction(self):
        return self.randDrive.next()
        #return self.randDrive.next()*np.array(self.simulationParams['randomActionScale'])
        #self.randDrive.multiGenNoise(50)
        #self.sinActionT = self.sinActionT+np.random.normal([0.1,0.5],[0.01,0.2])
        #return np.sin(self.sinActionT)
