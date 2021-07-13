import rospy
import numpy as np
import torch
import pybullet as p
import matplotlib.pyplot as plt

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

class LocalFrictionmapSensor:
    """
    Same sensor as the local heightmap sensor, but for the friction map.
    Give the option to return GT friction or a colormap.
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='local_heightmap'):
        # set up robot sensing parameters
        self.senseParams = {"senseDim":[5, 5], # width (m) and height (m) of terrain map
                            "senseResolution":[128, 128], # array giving resolution of map output (num pixels wide x num pixels high)
                            "sensorPose":[[0,0,0],[0,0,0,1]], # pose of sensor relative to body
                            "cmap":"" #Colormap to convert to (no cmap by default)
                            }
        self.senseParams.update(senseParamsIn)
        self.env = env
        self.is_time_series = False
        self.topic = topic
        self.cmap = plt.get_cmap(self.senseParams["cmap"]) if not self.senseParams["cmap"]=="" else None
        self.N = [4] + self.senseParams["senseResolution"] if self.cmap else [1] + self.senseParams["senseResolution"]

    def measure(self):
        robotPose = self.env.robot.getPositionOrientation()
        sensorAbsolutePose = p.multiplyTransforms(robotPose[0],robotPose[1],self.senseParams["sensorPose"][0],self.senseParams["sensorPose"][1])
        heightmap = self.env.terrain.sensedFrictionMap(sensorAbsolutePose,self.senseParams["senseDim"],self.senseParams["senseResolution"])

        if self.cmap is None:
            heightmap = torch.tensor(heightmap).float().unsqueeze(0) #Good practice to prepend a null channel dim
        else:
            heightmap = (heightmap - self.env.terrain.terrainParams["frictionOffset"]) / (self.env.terrain.terrainParams["frictionScale"]/1.5)
            heightmap = np.moveaxis(self.cmap(heightmap), [0, 1, 2], [1, 2, 0])
            heightmap = torch.tensor(heightmap).float()

        heightmap[heightmap.isnan()] = -1. #No nans

        return heightmap

    def to_rosmsg(self, data):
        msg = GridMap()
        msg.info.header.stamp = rospy.Time.now()
        msg.info.header.frame_id = "map"
        msg.info.resolution = self.senseParams["senseDim"][0] / self.senseParams["senseResolution"][0]
        msg.info.length_x = self.senseParams["senseDim"][0]
        msg.info.length_y = self.senseParams["senseDim"][1]
        msg.layers.append("height")
        data = data.numpy()
        data_msg = Float32MultiArray()
        data_msg.layout.dim = [MultiArrayDimension("column_index", data.shape[0], data.shape[0] * data.dtype.itemsize), MultiArrayDimension("row_index", data.shape[1], data.shape[1] * data.dtype.itemsize)]
        data_msg.data = data.reshape([1, -1])[0].tolist()
        msg.data = [data_msg]
        return msg
