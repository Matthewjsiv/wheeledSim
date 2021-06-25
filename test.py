from wheeledSim.envs.pybullet_sim import WheeledSimEnv
import time

env = WheeledSimEnv('configurations/latentSensorParams.yaml', render=True)
for i in range(3):
    env.reset()
    time.sleep(5)
    
input("done")
