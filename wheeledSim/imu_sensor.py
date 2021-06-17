import torch
import pybullet as p

from sensor_msgs.msg import IMU
from geometry_msgs.msg import Vector3

class IMUSensor:
    """
    Class to get IMU data from the clifford robot.
    organized as angular velocity, then linear acceleration
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='shock_travel'):
        self.env = env
        self.robot = env.robot
        self.physics_client_id = physics_client_id
        self.is_time_series=True
        self.N = [6, ]
        self.topic = topic
        self.last_linear_velocity = torch.zeros(3)
        self.last_position = torch.zeros(3)

    def measure(self):
        state = p.getLinkState(self.robot.cliffordID, self.robot.linkNameToID['body'], physicsClientId=self.physics_client_id, computeLinkVelocity=True)
        position = torch.tensor(state[0])
        linear_velocity = torch.tensor(state[-2])
        angular_velocity = torch.tensor(state[-1])

        linear_acceleration = (linear_velocity - self.last_linear_velocity) / self.env.timeStep

        #Check for simulator errors / env reset
        if torch.linalg.norm(position, self.last_position) > 0.1:
            linear_acceleration = torch.zeros(3)

        self.last_linear_velocity = linear_velocity
        self.last_position = position

        return torch.cat([angular_velocity, linear_acceleration]).float()

    def to_rosmsg(self, data):
        msg = IMU()
        msg.angular_velocity.x = data[0]
        msg.angular_velocity.y = data[1]
        msg.angular_velocity.z = data[2]
        msg.linear_acceleration.x = data[3]
        msg.linear_acceleration.y = data[4]
        msg.linear_acceleration.z = data[5]
        msg.orientation_covariance[0] = -1 #Per the standard in the rosmsg.
        return msg
