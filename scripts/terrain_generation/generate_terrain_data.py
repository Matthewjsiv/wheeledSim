import pybullet as p
import time
import numpy as np
import torch
import argparse
import os
import yaml
from wheeledSim.terrain.randomTerrain import randomRockyTerrain

def generate_terrain(config, physicsClientId, viz):
    """
    Key difference from regular config files is that we need to sample from an exponential distribution
    """
    terrain = randomRockyTerrain(config['terrainMapParams'])

    param_distribution = {}
    for k, v in config['terrainParams'].items():
        if isinstance(v, float):
            param_distribution[k] = v
        else:
            if v['distribution'] == 'exponential':
                param_distribution[k] = v['offset'] + np.random.exponential(v['scale'])
            elif v['distribution'] == 'uniform':
                param_distribution[k] = np.random.uniform(v['min'], v['max'])

    print(param_distribution)
    terrain.generate(param_distribution)

    assert abs(config['terrainMapParams']['widthScale'] - config['terrainMapParams']['heightScale']) < 1e-4, 'width and height scale have to match'
    width = torch.tensor(
        config['terrainMapParams']['mapWidth'] * config['terrainMapParams']['widthScale']
    )
    height = torch.tensor(
        config['terrainMapParams']['mapHeight'] * config['terrainMapParams']['heightScale']
    )
    res = torch.tensor(config['terrainMapParams']['heightScale'])

    #Assume ego-position is the bottom-middle of the map.
    origin = torch.Tensor([0., -height/2.])
    metadata = {
        'width' :width,
        'height':height,
        'resolution':res,
        'origin':origin
    }

    data = {
        'metadata':metadata,
        'data':torch.Tensor(terrain.gridZ).float()
    }

    if args.viz:
        input('Enter for next terrain')

    p.removeBody(terrain.terrainBody, physicsClientId=physicsClientId)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True, help="Path to dir containing terrain config files")
    parser.add_argument('--save_to', type=str, required=True, help="Directory to save heightmaps in")
    parser.add_argument('--n', type=int, required=True, help="Number of heightmaps to generate (n/k per config file, where k=# config files in dir)")
    parser.add_argument('--viz', type=int, required=False, default=0, help="1 to visualize each heightmap after creating it (not recommended for anything that isn't debugging)")
    args = parser.parse_args()

    dst_dir_exists = os.path.isdir(args.save_to)    
    if dst_dir_exists:
        x = input("{} exists. 'q' to quit, anything else to keep going.\n".format(args.save_to))
        if x.lower() == 'q':
            exit(0)
    else:
        os.mkdir(args.save_to)

    config_basenames = os.listdir(args.config_dir)

    physicsClientId = p.connect(p.GUI if args.viz else p.DIRECT)
    p.setGravity(0,0,-10)

    for i in range(args.n):
        config_basename = config_basenames[i % len(config_basenames)]
        config_fp = os.path.join(args.config_dir, config_basename)
        print('{}/{} ({})'.format(i+1, args.n, config_basename))
        config = yaml.safe_load(open(config_fp, 'r'))
        data = generate_terrain(config, physicsClientId, args.viz)
        data_fp = os.path.join(args.save_to, 'hmap_{}.pt'.format(i+1))
        torch.save(data, data_fp)
