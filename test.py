import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from wheeledSim.envs.pybullet_sim import WheeledSimEnv

env = WheeledSimEnv('configurations/testFricmapSensor.yaml', render=True)


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.show(block=False)

tmap_params = env.env.terrain.terrainMapParams
xmin = -tmap_params['mapWidth'] * tmap_params['widthScale'] / 2
xmax = tmap_params['mapWidth'] * tmap_params['widthScale'] / 2
ymin = -tmap_params['mapHeight'] * tmap_params['heightScale'] / 2
ymax = tmap_params['mapHeight'] * tmap_params['heightScale'] / 2

for i in range(3):
    obs = env.reset()

    for ii in range(100):
        act = [0.5, 0.5]
        obs, r, t, i = env.step(act)

        pos = obs['state'][:2]

        fmap = obs['frictionmap']

        if fmap.shape[0] == 1:
            fmap = fmap[0]
        else:
            fmap = fmap.permute(1, 2, 0)
        
        for ax in axs:
            ax.cla()

        axs[0].imshow(env.env.terrain.frictionMap, cmap='RdYlGn', extent=(xmin, xmax, ymin, ymax), vmin=0., vmax=1., origin='lower')
        axs[0].scatter(pos[0], pos[1], marker='x', c='r')
        axs[1].imshow(fmap, origin='lower')
        plt.pause(1e-1)
    
input("done")
