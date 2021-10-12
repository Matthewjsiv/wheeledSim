import torch
import pybullet as p

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from common_msgs.msg import ImuArray

class IMUSensor:
    """
    Class to get IMU data from the clifford robot.
    organized as angular velocity, then linear acceleration
    Implement a low-pass filter to make data less noisy.
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='imu'):
        self.senseParams = {
                "useVelocity":True,
                "useAngularVelocity":True,
                "useAcceleration":True,
                "alpha":0.005, # alpha param of the low-pass filter
                "sigma":0.01 # Noise param for sensor
                            }
        self.senseParams.update(senseParamsIn)
        self.env = env
        self.robot = env.robot
        self.physics_client_id = physics_client_id
        self.is_time_series=True
        self.N = [9, ]
        self.topic = topic
        self.last_linear_velocity = torch.zeros(3)
        self.last_position = torch.zeros(3)
        self.last_angular_velocity = torch.zeros(3)
        self.last_acceleration = torch.zeros(3)

    def measure(self):
        position, orientation = self.robot.getPositionOrientation()
        velocity = self.robot.getBaseVelocity_body()
        position = torch.tensor(list(position))
        linear_velocity = torch.tensor(velocity[:3])
        angular_velocity = torch.tensor(velocity[3:])

        linear_acceleration = (linear_velocity - self.last_linear_velocity) / self.env.timeStep

        #Check for simulator errors / env reset
        if torch.linalg.norm(position - self.last_position) > 0.1:
            linear_acceleration = torch.zeros(3)

        
        linear_acceleration = (1-self.senseParams['alpha'])*self.last_acceleration + self.senseParams['alpha']*linear_acceleration
        linear_acceleration_measurement = linear_acceleration + torch.randn(3)*self.senseParams['sigma']

        angular_velocity = (1-self.senseParams['alpha'])*self.last_angular_velocity + self.senseParams['alpha']*angular_velocity
        angular_velocity_measurement = angular_velocity + torch.randn(3)*self.senseParams['sigma']

        linear_velocity_measurement = linear_velocity + torch.randn(3)*self.senseParams['sigma']

        self.last_linear_velocity = linear_velocity
        self.last_angular_velocity = angular_velocity
        self.last_position = position
        self.last_acceleration = linear_acceleration

        return torch.cat([linear_velocity_measurement, angular_velocity_measurement, linear_acceleration_measurement]).float()

    def to_rosmsg(self, data):
        if len(data.shape) == 1:
            msg = Imu()
            #NOTE: I'm overloading the orientation part of IMU
            msg.orientation.x = data[0]
            msg.orientation.y = data[1]
            msg.orientation.z = data[2]
            msg.orientation.w = -1.
            msg.angular_velocity.x = data[3]
            msg.angular_velocity.y = data[4]
            msg.angular_velocity.z = data[5]
            msg.linear_acceleration.x = data[6]
            msg.linear_acceleration.y = data[7]
            msg.linear_acceleration.z = data[8]
            msg.orientation_covariance[0] = -1 #Per the standard in the rosmsg.
            return msg

        elif len(data.shape) == 2:
            msg = ImuArray()
            msg.imus = [self.to_rosmsg(x) for x in data]
            return msg

        else:
            print('Got more than 2 dims to parse for IMU')
            return Imu()
