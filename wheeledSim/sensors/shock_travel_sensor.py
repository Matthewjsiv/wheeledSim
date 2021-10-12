import torch
import pybullet as p

from geometry_msgs.msg import Point
from common_msgs.msg import ShockTravel, ShockTravelArray

class ShockTravelSensor:
    """
    Class to get shock travel from the clifford robot.
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='shock_travel'):
        self.robot = env.robot
        self.physics_client_id = physics_client_id
        self.is_time_series=True
        self.N = [4, ]
        self.topic = topic

    def measure(self):
        frsupper_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['frsupper'], physicsClientId=self.physics_client_id)[0])
        frslower_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['frslower'], physicsClientId=self.physics_client_id)[0])
        flsupper_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['flsupper'], physicsClientId=self.physics_client_id)[0])
        flslower_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['flslower'], physicsClientId=self.physics_client_id)[0])
        brsupper_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['brsupper'], physicsClientId=self.physics_client_id)[0])
        brslower_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['brslower'], physicsClientId=self.physics_client_id)[0])
        blsupper_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['blsupper'], physicsClientId=self.physics_client_id)[0])
        blslower_pos = torch.tensor(p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['blslower'], physicsClientId=self.physics_client_id)[0])

        fr_travel = torch.linalg.norm(frsupper_pos - frslower_pos)
        fl_travel = torch.linalg.norm(flsupper_pos - flslower_pos)
        br_travel = torch.linalg.norm(brsupper_pos - brslower_pos)
        bl_travel = torch.linalg.norm(blsupper_pos - blslower_pos)

        return torch.tensor([fl_travel, fr_travel, bl_travel, br_travel]).float()

    def to_rosmsg(self, data):
        if len(data.shape) == 1:
            msg = ShockTravel()
            msg.front_left = data[0].item()
            msg.front_right = data[1].item()
            msg.rear_left = data[2].item()
            msg.rear_right = data[3].item()
            return msg
        elif len(data.shape) == 2:
            msg = ShockTravelArray()
            msg.shock_travels = [self.to_rosmsg(x) for x in data]
            return msg
        else:
            print('Got more than 2 dims to parse for Shock Travel')
            return Imu()
