from wheeledSim.sensors.front_camera_sensor import FrontCameraSensor
from wheeledSim.sensors.lidar_sensor import LidarSensor
from wheeledSim.sensors.imu_sensor import IMUSensor
from wheeledSim.sensors.shock_travel_sensor import ShockTravelSensor
from wheeledSim.sensors.local_heightmap_sensor import LocalHeightmapSensor
from wheeledSim.sensors.local_frictionmap_sensor import LocalFrictionmapSensor
from wheeledSim.sensors.global_map_sensor import GlobalmapSensor
from wheeledSim.sensors.steer_angle_sensor import SteerAngleSensor

sensor_str_to_obj = {
    'FrontCameraSensor':FrontCameraSensor,
    'LidarSensor':LidarSensor,
    'IMUSensor':IMUSensor,
    'ShockTravelSensor':ShockTravelSensor,
    'LocalHeightmapSensor':LocalHeightmapSensor,
    'LocalFrictionmapSensor':LocalFrictionmapSensor,
    'GlobalmapSensor':GlobalmapSensor,
    'SteerAngleSensor':SteerAngleSensor
}
