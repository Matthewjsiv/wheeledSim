import rospy
import numpy as np
import torch
import pybullet as p

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class FrontCameraSensor:
    """
    Sensor that gets local heightmaps. Needs the robot to get pose, and terrain.
    For sense params, need pose to the robot body, the size and the resolution.
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='front_cam'):
        # set up robot sensing parameters
        self.senseParams = {"FOV":45.0, # camera FOV
                            "aspect":1.0, # camera aspect
                            "senseDepth":[0.1, 18.1], #min/max distance to register pixels (m)
                            "senseResolution":[128, 128], #width/height of the image
                            "camDist": .11 #distance from camera point to render camera image
                            }

        self.senseParams.update(senseParamsIn)
        self.env = env
        self.is_time_series = False
        self.N = [self.senseParams["senseResolution"][0],self.senseParams["senseResolution"][1],5]
        self.topic = topic
        self.physicsClientId = physics_client_id

    def measure(self):
        pose = self.env.robot.getPositionOrientation()
        # pose = self.env.getPositionOrientation()
        pose = p.multiplyTransforms(pose[0],pose[1],self.senseParams["sensorPose"][0],self.senseParams["sensorPose"][1])

        posx,posy,posz = pose[0][0],pose[0][1],pose[0][2]

        rotation = p.getMatrixFromQuaternion(pose[1])
        forwardDir = [rotation[0], rotation[3], rotation[6]]
        upDir = [rotation[2], rotation[5], rotation[8]]

        # posx += forwardDir[0]*.12
        # posy += forwardDir[1]*.12
        # posz += forwardDir[2]*.12
        # posx += upDir[0]*.12
        # posy += upDir[1]*.12
        # posz += upDir[2]*.12

        q = pose[1]

        rollAngle = np.arctan2(2.0 * (q[2]*q[1] + q[3]*q[0]),1.0 - 2.0*(q[0]*q[0] + q[1]*q[1]))*180/np.pi
        #angle down a little bit
        pitchAngle = -1*np.arcsin(2.0 * (q[1]*q[3] - q[2]*q[0]))*180/np.pi - self.senseParams["sensorAngle"]
        headingAngle = np.arctan2(2.0 * (q[2]*q[3] + q[0]*q[1]), -1.0 + 2.0*(q[3]*q[3] + q[0]*q[0]))*180/np.pi - 90

        view_matrix = p.computeViewMatrixFromYawPitchRoll((posx,posy,posz),self.senseParams["camDist"],headingAngle,pitchAngle,rollAngle,2,physicsClientId=self.physicsClientId)
        projectionMatrix = p.computeProjectionMatrixFOV(fov=self.senseParams["FOV"],
            aspect=self.senseParams["aspect"],
            nearVal=self.senseParams["senseDepth"][0],
            farVal=self.senseParams["senseDepth"][1])
        # p.resetDebugVisualizerCamera(1.0,headingAngle,-15,pos,physicsClientId=self.physicsClientId)

        w,h,rgbImg,depthImg,segImg = p.getCameraImage(
                self.senseParams["senseResolution"][0],
                self.senseParams["senseResolution"][1],
                view_matrix,
                projectionMatrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_NO_SEGMENTATION_MASK,
                physicsClientId=self.physicsClientId)

        fullImg = np.dstack((rgbImg,depthImg))

        torch_img = torch.tensor(fullImg).float()
        torch_img[:, :, :3] /= 255. #[0-1 scaling better for learning]
        return torch_img

    def to_rosmsg(self, data):
        msg = GridMap()
        msg.info.header.stamp = rospy.Time.now()
        msg.info.header.frame_id = "robot"
        msg.info.length_x = self.senseParams["senseResolution"][0]
        msg.info.length_y = self.senseParams["senseResolution"][1]
        msg.layers.append("RGBAD")
        data = data.numpy()
        data_msg = Float32MultiArray()
        data_msg.layout.dim = [MultiArrayDimension("column_index", data.shape[0], data.shape[0] * data.dtype.itemsize), MultiArrayDimension("row_index", data.shape[1], data.shape[1] * data.dtype.itemsize), MultiArrayDimension("channel_index", data.shape[2], data.shape[2] * data.dtype.itemsize)]
        data_msg.data = data.reshape([1, -1])[0].tolist()
        msg.data = [data_msg]

        return msg
