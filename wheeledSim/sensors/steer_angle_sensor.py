import rospy
import torch
import pybullet as p

from common_msgs.msg import Float32Stamped

class SteerAngleSensor:
    """
    Get the current steering angle
    """
    def __init__(self, env, senseParamsIn={}, physics_client_id=0, topic='steering_angle'):
        self.robot = env.robot
        self.physics_client_id = physics_client_id
        self.is_time_series=False
        self.N = [1, ]
        self.topic = topic

    def measure(self):
        joint_states = p.getJointStates(self.robot.cliffordID, [self.robot.jointNameToID['axle2frwheel'], self.robot.jointNameToID['axle2flwheel']], physicsClientId=self.physics_client_id)
        steer_angle = 0.5 * (joint_states[0][0] + joint_states[1][0])

        return torch.tensor([steer_angle]).float()

    def to_rosmsg(self, data):
        msg = Float32Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = data
        return msg
