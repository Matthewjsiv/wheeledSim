import torch
import pybullet as p

from geometry_msgs.msg import Point

class ShockTravelSensor:
    """
    Class to get shock travel from the clifford robot.
    """
    def __init__(self, env, physics_client_id=0, topic='shock_travel'):
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
        #TODO: implement
        return Point()
