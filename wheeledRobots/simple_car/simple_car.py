import pybullet as p
import numpy as np
import os

# This class adds a clifford (wheeled off road) robot to a given PyBullet simulation
class SimpleCar:
    def __init__(self,t_params, sdfRootPath=None,physicsClientId=0, params={}):
        if sdfRootPath is None:
            sdfRootPath = os.path.abspath(os.path.join(os.path.realpath(__file__),'../'))
        # this is the folder that the file clifford.sdf is in (relative to the folder this script is in)
        self.sdfPath = os.path.abspath(os.path.join(sdfRootPath,'simple_car.urdf'))
        # define which PyBullet simulation to use
        self.physicsClientId=physicsClientId
        # Default parameters to use with Clifford
        # self.traction = 0
        self.params = {"maxThrottle":50,
                        "maxSteerAngle":0.5,
                        "susOffset":-0.01,
                        "susLowerLimit":-0.005,
                        "susUpperLimit":0.008,
                        "susDamping":10,
                        "susSpring":100,
                        "traction":10.0,
                        "massScale":1.0,
                        "tireMassScale":1.0,
                        "fixedSuspension":False,
                        "frictionMap": np.ones([512,512]),
                        "front_left_wheel_link": 1.5,
                        "front_right_wheel_link": 1.5,
                        "back_left_wheel_link":1.5,
                        "back_right_wheel_link":1.5}
        self.mapWidth = t_params["mapWidth"]
        self.mapHeight = t_params["mapHeight"]
        self.widthScale = t_params["widthScale"]
        self.heightScale = t_params["heightScale"]
        # change default params if defined
        for param in self.params:
            if param in params:
                self.params[param] = params[param]
        # set up Clifford robot in simulation

        self.importClifford()

    def importClifford(self):
        # load sdf file (this file defines open chain clifford. need to add constraints to make closed chain)
        self.cliffordID = p.loadURDF(self.sdfPath,physicsClientId=self.physicsClientId)
        # define number of joints of clifford robot
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        initialJointStates = p.getJointStatesMultiDof(self.cliffordID,range(nJoints),physicsClientId=self.physicsClientId)
        self.initialJointPositions = [initialJointStates[i][0] for i in range(nJoints)]
        self.initialJointVelocities = [initialJointStates[i][1] for i in range(nJoints)]
        self.buildModelDict()

        self.loosenModel()
        self.changeTraction()
#        self.changeColor()
        self.reset()

    def reset(self,pose=[[0,0,0.3],[0,0,0,1]]):
        p.resetBasePositionAndOrientation(self.cliffordID, pose[0],pose[1],physicsClientId=self.physicsClientId)
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)

        reset_positions = [self.initialJointPositions[i] for i in self.resetJointsIDs]
        reset_velocities = [self.initialJointVelocities[i] for i in self.resetJointsIDs]
        p.resetJointStatesMultiDof(self.cliffordID,self.resetJointsIDs,reset_positions,reset_velocities,physicsClientId=self.physicsClientId)

    def buildModelDict(self):
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        self.jointNameToID = {}
        self.linkNameToID = {}
        for i in range(nJoints):
            JointInfo = p.getJointInfo(self.cliffordID,i,physicsClientId=self.physicsClientId)
            self.jointNameToID[JointInfo[1].decode('UTF-8')] = JointInfo[0]
            self.linkNameToID[JointInfo[12].decode('UTF-8')] = JointInfo[0]

        measuredJointNames = ['steer_joint', 'front_left_wheel_joint', 'front_right_wheel_joint', 'back_left_wheel_joint', 'back_right_wheel_joint']
        self.measuredJointIDs = [self.jointNameToID[name] for name in measuredJointNames]
        self.motorJointsIDs = [self.jointNameToID[name] for name in measuredJointNames[-4:]]
        self.resetJointsIDs = [i for i in range(nJoints) if len(self.initialJointPositions[i]) > 0]

    def changeColor(self,color=None):
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        if color is None:
            color = [0.6,0.1,0.1,1]
        for i in range(-1,nJoints):
            p.changeVisualShape(self.cliffordID,i,rgbaColor=color,specularColor=color,physicsClientId=self.physicsClientId)
        tires = ['front_left_wheel_link','front_right_wheel_link','back_left_wheel_link','back_right_wheel_link']
        for tire in tires:
            p.changeVisualShape(self.cliffordID,self.linkNameToID[tire],rgbaColor=[0.15,0.15,0.15,1],specularColor=[0.15,0.15,0.15,1],physicsClientId=self.physicsClientId)

    def loosenModel(self):
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        tireIndices = [self.linkNameToID[name] for name in ['front_left_wheel_link','front_right_wheel_link','back_left_wheel_link','back_right_wheel_link']]
        for i in range(nJoints):
            if len(p.getJointStateMultiDof(bodyUniqueId=self.cliffordID,jointIndex=i,physicsClientId=self.physicsClientId)[0]) == 4 \
            and not self.params["fixedSuspension"]:
                p.setJointMotorControlMultiDof(bodyUniqueId=self.cliffordID,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=[0,0,0,1],
                                                positionGain=0,
                                                velocityGain=0,
                                                force=[0,0,0],physicsClientId=self.physicsClientId)
            dynamicsData = p.getDynamicsInfo(self.cliffordID,i,physicsClientId=self.physicsClientId)
            massScale = self.params["massScale"]
            if i in tireIndices:
                massScale = massScale*self.params['tireMassScale']
            newMass = dynamicsData[0]*massScale
            newInertia = [dynamicsData[2][j]*massScale for j in range(len(dynamicsData[2]))]
            p.changeDynamics(self.cliffordID,i,mass = newMass, localInertiaDiagonal=newInertia,linearDamping=0.2,angularDamping=0.2,restitution=0,physicsClientId=self.physicsClientId)

    def changeTraction(self,newTraction=None):
        if newTraction!=None:
            self.params["traction"] = newTraction
        tires = ['front_left_wheel_link','front_right_wheel_link','back_left_wheel_link','back_right_wheel_link']
        for tire in tires:
            p.changeDynamics(self.cliffordID,self.linkNameToID[tire],lateralFriction=self.params["traction"],physicsClientId=self.physicsClientId)

    def changeTireTraction(self,tire,newTraction):
        # print(tire + '---->' + str(newTraction))
        self.params[tire] = newTraction
        p.changeDynamics(self.cliffordID,self.linkNameToID[tire],lateralFriction=newTraction,physicsClientId=self.physicsClientId)

    def updateTraction(self):
        pwb,rwb = self.getPositionOrientation()
        x= int(pwb[0]/self.widthScale + self.mapWidth/2)
        y = int(pwb[1]/self.heightScale + self.mapHeight/2)
        fm = self.params["frictionMap"]
        newtract = fm[y,x]
        if newtract != self.params["traction"]:
            self.changeTraction(newtract)
#            print(self.params["traction"])

    def updateTraction4Tire(self):
        #order should allegedly be
        tlist = ['front_left_wheel_link','front_right_wheel_link','back_left_wheel_link','back_right_wheel_link']
        poses = self.getTirePos()
        fm = self.params["frictionMap"]

        #width and height might need to be switched - not sure atm bc it's square
        for t in range(4):
            x = int(poses[t][0][0]/self.widthScale + self.mapWidth/2)
            y = int(poses[t][0][1]/self.heightScale + self.mapHeight/2)
            newtract = fm[y,x]
            if newtract != self.params[tlist[t]]:
                self.changeTireTraction(tlist[t],newtract)
#                print(tlist[t] + ': ' + str(self.params[tlist[t]]))

    def updateSpringForce(self):
        pass

    def drive(self,driveSpeed):
        maxForce = 100
        p.setJointMotorControlArray(self.cliffordID,self.motorJointsIDs,p.VELOCITY_CONTROL,
                                    targetVelocities=[driveSpeed*self.params["maxThrottle"]]*4,
                                    forces=[maxForce]*4,
                                    physicsClientId=self.physicsClientId)
    def steer(self,angle):
        maxForce = 10000
        p.setJointMotorControl2(bodyUniqueId=self.cliffordID,
        jointIndex=self.jointNameToID['front_right_bar_joint'],
        controlMode=p.POSITION_CONTROL,
        maxVelocity = 10,
        targetPosition = angle*self.params["maxSteerAngle"],
        force = maxForce,physicsClientId=self.physicsClientId)

        p.setJointMotorControl2(bodyUniqueId=self.cliffordID,
        jointIndex=self.jointNameToID['front_left_bar_joint'],
        controlMode=p.POSITION_CONTROL,
        maxVelocity = 10,
        targetPosition = angle*self.params["maxSteerAngle"],
        force = maxForce,physicsClientId=self.physicsClientId)

        #p.setJointMotorControl2(bodyUniqueId=self.cliffordID,
        #jointIndex=self.jointNameToID['axle2flwheel'],
        #controlMode=p.POSITION_CONTROL,
        #targetPosition = angle,
        #force = maxForce,physicsClientId=self.physicsClientId)
    def getBaseVelocity_body(self):
        gwb = p.getBasePositionAndOrientation(self.cliffordID,physicsClientId=self.physicsClientId)
        Rbw = p.invertTransform(gwb[0],gwb[1])[1]
        Vw = p.getBaseVelocity(self.cliffordID,physicsClientId=self.physicsClientId)
        v_b = p.multiplyTransforms([0,0,0],Rbw,Vw[0],[0,0,0,1])[0]
        w_b = p.multiplyTransforms([0,0,0],Rbw,Vw[1],[0,0,0,1])[0]
        return list(v_b)+list(w_b)
    def getPositionOrientation(self):
        pwb,Rwb = p.getBasePositionAndOrientation(self.cliffordID,physicsClientId=self.physicsClientId)
        #forwardDir = p.multiplyTransforms([0,0,0],Rwb,[1,0,0],[0,0,0,1])[0]
        #headingAngle = np.arctan2(forwardDir[1],forwardDir[0])
        #Rbw = p.invertTransform([0,0,0],Rwb)[1]
        #upDir = p.multiplyTransforms([0,0,0],Rbw,[0,0,1],[0,0,0,1])[0]
        #tiltAngles = [np.arccos(upDir[2])]
        #tiltAngles.append(np.arctan2(upDir[1],upDir[0]))
        return (pwb,Rwb)#,headingAngle,tiltAngles)
    def measureJoints(self):
        jointStates = p.getJointStates(self.cliffordID,self.measuredJointIDs,physicsClientId=self.physicsClientId)
        positionReadings = [jointStates[i][0] for i in range(5)]
        velocityReadings = [jointState[1] for jointState in jointStates]
        measurements = positionReadings + velocityReadings
        return measurements
    def getTirePos(self):
        # print(self.linkNameToID)
        tires = ['front_left_wheel_link','front_right_wheel_link','back_left_wheel_link','back_right_wheel_link']
        id_list = []
        for tire in tires:
            id_list.append(self.linkNameToID[tire])
        linkStates = p.getLinkStates(self.cliffordID,id_list,physicsClientId=self.physicsClientId)
        return linkStates
