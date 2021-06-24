import rospy
import numpy as np
import torch
import pybullet as p

# from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

class LidarSensor:
    """
    Sensor that gets local heightmaps. Needs the robot to get pose, and terrain.
    For sense params, need pose to the robot body, the size and the resolution.
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='lidar'):
        # set up robot sensing parameters

        self.senseParams = {"senseDim":[2.*np.pi,np.pi/4.], # angular width and height of lidar sensing
                        "lidarAngleOffset":[0,0], # offset of lidar sensing angle
                        "lidarRange":120, # max distance of lidar sensing
                        "senseResolution":[512,16], # resolution of sensing (width x height)
                        "removeInvalidPointsInPC":False, # remove invalid points in point cloud
                        "senseType":2,
                        "sensorPose":[[0,0,0.3],[0,0,0,1]]} # pose of sensor relative to body

        self.senseParams.update(senseParamsIn)
        self.env = env
        self.is_time_series = False
        self.N = [self.senseParams["senseResolution"][0] * self.senseParams["senseResolution"][1], 3] if self.senseParams['senseType'] != 1 else self.senseParams['senseResolution']
        self.topic = topic
        self.physicsClientId = physics_client_id

    def measure(self):
        robotPose = self.env.robot.getPositionOrientation()
        # robotPose = self.env.getPositionOrientation()
        sensorAbsolutePose = p.multiplyTransforms(robotPose[0],robotPose[1],self.senseParams["sensorPose"][0],self.senseParams["sensorPose"][1])
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
        if self.senseParams["senseType"] == 1: # return depth map
            sensorData = rangeData
        else: # return point cloud
            lidarPoints = [rayResults[i][3] for i in range(len(rayResults))]
            lidarPoints = np.array(lidarPoints).transpose()
            if self.senseParams["removeInvalidPointsInPC"]:
                lidarPoints = lidarPoints[:,rangeData.reshape(-1)<1]
            sensorData = lidarPoints
        #shape = (3, 8192) before transpose
        sensorData = sensorData.T

        return torch.tensor(sensorData).float()



    def to_rosmsg(self, data):
        data = data.numpy()

        msg = PointCloud2()

        msg.header.stamp = rospy.Time().now()

        msg.header.frame_id = "map"

        if len(data.shape) == 3:
            msg.height = data.shape[1]
            msg.width = data.shape[0]
        else:
            msg.height = 1
            msg.width = len(data)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * data.shape[0]
        # msg.is_dense = int(np.isfinite(points).all())
        msg.is_dense = False
        msg.data = np.asarray(data, np.float32).tostring()


        return msg
