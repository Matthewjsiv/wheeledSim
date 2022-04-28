import rospy
import numpy as np
import torch
import pybullet as p

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

class GlobalmapSensor:
    """
    Sensor that gets the entire global map and returns it as a gridmap
    This is done because we're testing all our map stuff in the global frame.
    Currently, take:
        1. Height
        2. RGB channels
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='global_map'):
        # set up robot sensing parameters
        self.senseParams = {
                            }
        self.senseParams.update(senseParamsIn)
        self.env = env
        self.is_time_series = False
        self.N = [4, self.env.terrain.terrainMapParams["mapWidth"], self.env.terrain.terrainMapParams["mapHeight"]]
        self.topic = topic

    def measure(self):
        heightmap = torch.tensor(self.env.terrain.gridZ).unsqueeze(-1)
        colormap = torch.tensor(self.env.terrain.colormap)
        mapdata = torch.cat([heightmap, colormap], dim=-1).movedim(-1, 0)
        return mapdata

    def to_rosmsg(self, data):
        layers = ["elevation", "red", "blue", "green"]

        msg = GridMap()
        msg.info.header.stamp = rospy.Time.now()
        msg.info.header.frame_id = "world"
        msg.info.resolution = self.env.terrain.terrainMapParams["widthScale"]
        msg.info.length_x = self.env.terrain.terrainMapParams["mapWidth"] * msg.info.resolution
        msg.info.length_y = self.env.terrain.terrainMapParams["mapHeight"] * msg.info.resolution
        msg.layers = layers
        data = data.numpy().astype(np.float32)
        for i, layer in enumerate(layers):
            data_msg = Float32MultiArray()
            subdata = data[i, ::-1, ::-1] #gridmap starts from the top
            data_msg.layout.dim = [MultiArrayDimension("column_index", subdata.shape[0], subdata.shape[1] * data.dtype.itemsize), MultiArrayDimension("row_index", subdata.shape[1], subdata.shape[1] * data.dtype.itemsize)]
            data_msg.data = subdata.reshape([1, -1])[0].tolist()
            msg.data.append(data_msg)
        return msg
