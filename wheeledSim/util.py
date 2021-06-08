"""
Collection of utility functions
"""

import numpy as np

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
