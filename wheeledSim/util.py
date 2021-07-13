"""
Collection of utility functions
"""
import os
import numpy as np

def maybe_mkdir(fp, force=True):
    if not os.path.exists(fp):
        os.mkdir(fp)
    elif not force:
        x = input('{} already exists. Hit enter to continue and overwrite. Q to exit.'.format(fp))
        if x.lower() == 'q':
            exit(0)

def quat_to_yaw(q):
    """
    Extract yaw from a quaternion (return as numpy)
    """
    qw = q.w
    qx = q.x
    qy = q.y
    qz = q.z
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    return yaw
