import rospy
import pybullet as p
import numpy as np
import torch

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from wheeledSim.randomTerrain import *
from wheeledSim.randomExplorationPolicy import *
from wheeledSim.simController import simController

class rosSimController(simController):
    """
    Same as simcontroller, but is continuously running and is integrated to use rostopics.
    """
    def __init__(self,robot,physicsClientId=0,simulationParamsIn={},senseParamsIn={},
                terrainMapParamsIn={},terrainParamsIn={},explorationParamsIn={},sensors=[]):
        super(rosSimController, self).__init__(robot, physicsClientId, simulationParamsIn, senseParamsIn, terrainMapParamsIn, terrainParamsIn, explorationParamsIn, sensors)
        self.curr_drive = np.array([0., 0.])
        self.buf = {k:[torch.zeros(*k.N)] * self.stepsPerControlLoop for k in self.sensors if k.is_time_series}

    def set_sensors(self, sensors):
        self.sensors = sensors
        self.buf = {k:[torch.zeros(*k.N)] * self.stepsPerControlLoop for k in self.sensors if k.is_time_series}

    def handle_cmd(self, msg):
        throttle = msg.drive.speed
        steer = msg.drive.steering_angle
        self.curr_drive = np.array([throttle, steer])

    def get_sensing(self):
        state, sensing, action = self.state

        odom_msg = self.state_to_odom(state)
        sensing_msgs = {k:k.to_rosmsg(v) for k,v in sensing.items()}
        return odom_msg, sensing_msgs

    def state_to_odom(self, state):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = state[0]
        msg.pose.pose.position.y = state[1]
        msg.pose.pose.position.z = state[2]
        msg.pose.pose.orientation.x = state[3]
        msg.pose.pose.orientation.y = state[4]
        msg.pose.pose.orientation.z = state[5]
        msg.pose.pose.orientation.w = state[6]
        msg.twist.twist.linear.x = state[7]
        msg.twist.twist.linear.y = state[8]
        msg.twist.twist.linear.z = state[9]
        msg.twist.twist.angular.x = state[10]
        msg.twist.twist.angular.y = state[11]
        msg.twist.twist.angular.z = state[12]
        return msg

    def step(self):
        """
        No args because we get everything from rostopics.
        """
        throttle = self.curr_drive[0]
        steering = self.curr_drive[1]
        # Create Prediction Input
        # check if last pose of robot has been recorded
        if not self.lastStateRecordFlag:
            self.lastPose = self.robot.getPositionOrientation()
            self.lastVel = self.robot.getBaseVelocity_body()
            if self.senseParams["recordJointStates"]:
                self.lastJointState = self.robot.measureJoints()
                self.lastAbsoluteState = list(self.lastPose[0])+list(self.lastPose[1])+self.lastVel[:] + self.lastJointState[:]
            else:
                self.lastAbsoluteState = list(self.lastPose[0])+list(self.lastPose[1])+self.lastVel[:]
        #simulate sensing (generate height map or lidar point cloud)
        sensingData = {k: torch.stack(self.buf[k], dim=0) if k.is_time_series else k.measure() for k in self.sensors}
        # store state-action for motion prediction
        stateActionData = [self.lastAbsoluteState,sensingData,self.curr_drive] #(absolute robot state, sensing data, action)
        # command robot throttle & steering and simulate
        self.robot.drive(throttle)
        self.robot.steer(steering)

        for i in range(self.stepsPerControlLoop):
            self.stepSim()
            for sensor in self.sensors:
                if sensor.is_time_series:
                    newpt = sensor.measure()
                    self.buf[sensor] = self.buf[sensor][1:] + [newpt]

        # Record outcome state
        newPose = self.robot.getPositionOrientation()
        # check how long robot has been stuck
        if (newPose[0][0]-self.lastX)*(newPose[0][0]-self.lastX) + (newPose[0][1]-self.lastY)*(newPose[0][1]-self.lastY)> self.moveThreshold:
            self.lastX = newPose[0][0]
            self.lastY = newPose[0][1]
            self.stopMoveCount = 0
        else:
            self.stopMoveCount +=1
        # relative position, body twist, joint position and velocity
        self.lastPose = newPose
        self.lastVel = self.robot.getBaseVelocity_body()

        """
        if self.senseParams["recordJointStates"]:
            self.lastJointState = self.robot.measureJoints()
            self.lastAbsoluteState = list(self.lastPose[0])+list(self.lastPose[1])+self.lastVel[:] + self.lastJointState[:]
        else:
            self.lastAbsoluteState = list(self.lastPose[0])+list(self.lastPose[1])+self.lastVel[:]
        self.lastStateRecordFlag = True
        newStateData = [self.lastAbsoluteState]
        """

        self.state = stateActionData

if __name__ == '__main__':
    from grid_map_msgs.msg import GridMap

    from wheeledRobots.clifford.cliffordRobot import Clifford

    from wheeledSim.shock_travel_sensor import ShockTravelSensor
    from wheeledSim.local_heightmap_sensor import LocalHeightmapSensor

    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    """initialize clifford robot"""
    cliffordParams={"maxThrottle":20, # dynamical parameters of clifford robot
                    "maxSteerAngle":0.5,
                    "susOffset":-0.00,
                    "susLowerLimit":-0.01,
                    "susUpperLimit":0.00,
                    "susDamping":10,
                    "susSpring":500,
                    "traction":1.25,
                    "massScale":1.0}
    terrainMapParams = {"mapWidth":200, # width of matrix
                    "mapHeight":200, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1,
                    "depthScale":1.0
                    }
    terrainParams = {"AverageAreaPerCell":1.0,
                    "cellPerlinScale":5,
                    "cellHeightScale":0.0, # parameters for generating terrain
                    "smoothing":0.7,
                    "perlinScale":2.5,
                    "perlinHeightScale":0.0,
                    }

    robot = Clifford(params=cliffordParams,physicsClientId=physicsClient)

    env = rosSimController(robot, terrainParamsIn=terrainParams, terrainMapParamsIn=terrainMapParams)
    env.set_sensors([LocalHeightmapSensor(env), ShockTravelSensor(env)])

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(10)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    map_pub = rospy.Publisher("/local_map", GridMap, queue_size=1)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()
        heightmap = sensing[env.sensors[0]]
        odom_pub.publish(state)
        map_pub.publish(heightmap)
        rate.sleep()
