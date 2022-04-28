#!/bin/bash

for i in {1..44}
do
    python3 generate_terrain_data.py --config_dir ../../configurations/terrain_generation/multi_perlin/ --n 250 --viz 0 --save_to /media/yamaha/4e369a85-4ded-4dcb-b666-9e0d521555c7/datasets/heightmap_prediction/dir_$i

done
