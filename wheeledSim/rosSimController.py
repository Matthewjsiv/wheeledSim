import rospy
import pybullet
import numpy as np
import torch
import yaml

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.terrain.randomTerrain import *
from wheeledSim.randomExplorationPolicy import *
from wheeledSim.simController import simController
from wheeledSim.sensors.sensor_map import sensor_str_to_obj

class rosSimController(simController):
    """
    Same as simcontroller, but is continuously running and is integrated to use rostopics.
    """
    def __init__(self, config_file, T=-1, render=True):

        # open file with configuration parameters
        stream = open(config_file, 'r')
        config = yaml.load(stream, Loader=yaml.FullLoader)

        self.client = pybullet.connect(pybullet.GUI, options=config.get('backgroundColor', "")) if render else pybullet.connect(pybullet.DIRECT)
        self.robot = Clifford(params=config['cliffordParams'], t_params = config['terrainMapParams'], physicsClientId=self.client)

        # load simulation environment
        super(rosSimController, self).__init__(self.robot, self.client, config['simulationParams'], config['senseParams'],
                                 config['terrainMapParams'], config['terrainParams'], config['explorationParams'])

        # initialize all sensors from config file
        sensors = []
        self.sense_dict = config.get('sensors', {})
        if self.sense_dict is None:
            self.sense_dict = {}

        for sensor in self.sense_dict.values():
            assert sensor['type'] in sensor_str_to_obj.keys(), "{} not a valid sensor type. Valid sensor types are {}".format(sensor['type']. sensor_str_to_obj.keys())

            sensor_cls = sensor_str_to_obj[sensor['type']]
            sensor = sensor_cls(self, senseParamsIn=sensor.get('params', {}), topic=sensor.get('topic', None))
            sensors.append(sensor)

        self.set_sensors(sensors)
        self.curr_drive = np.zeros(2)

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
        sensing_msgs = {k.topic:k.to_rosmsg(v) for k,v in sensing.items()}
        return odom_msg, sensing_msgs

    def state_to_odom(self, state):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.child_frame_id = "robot"
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
        sense_data = {k:torch.stack(self.buf[k], dim=0) if k.is_time_series else k.measure() for k in self.sensors}

        self.state = [self.lastAbsoluteState, sense_data, self.curr_drive]

if __name__ == '__main__':
    from grid_map_msgs.msg import GridMap

    from wheeledRobots.clifford.cliffordRobot import Clifford

    from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor
    from wheeledSim.sensors.local_heightmap_sensor import LocalHeightmapSensor
    from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
    from wheeledSim.sensors.lidar_sensor import LidarSensor

    env = rosSimController('../configurations/sysidEnvParams.yaml', render=True)

#    env = rosSimController(robot, terrainParamsIn=terrainParams, terrainMapParamsIn=terrainMapParams)
#    env.set_sensors([LocalHeightmapSensor(env), ShockTravelSensor(env),FrontCameraSensor(env),LidarSensor(env)])

    rospy.init_node("pybullet_simulator")
    rate = rospy.Rate(10)

    cmd_sub = rospy.Subscriber("/cmd", AckermannDriveStamped, env.handle_cmd, queue_size=1)
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    map_pub = rospy.Publisher("/local_map", GridMap, queue_size=1)
    cam_pub = rospy.Publisher("/front_camera", GridMap, queue_size=1)

    while not rospy.is_shutdown():
        env.step()
        state, sensing = env.get_sensing()
        heightmap = sensing['heightmap']
        img = sensing['front_camera']
        odom_pub.publish(state)
        map_pub.publish(heightmap)
        cam_pub.publish(img)
        rate.sleep()
