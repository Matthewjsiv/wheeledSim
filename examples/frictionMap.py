import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltmg
from PIL import Image
import cv2 as cv
from tifffile import imsave

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 50


def use_ogrid(w,h,sq):
    coords = np.ogrid[0:w, 0:h]
    idx = (coords[0] // sq + coords[1] // sq) % 2
    vals = np.array([2.5, 0])#, dtype=np.uint8)
    img = vals[idx]
    return img

fm = use_ogrid(512,512,40)

plt.imshow(fm)
# fm = np.ones([512,512])
# fm[:int(512/2),:] = 0
# fm = checkerboard((512,512))
# plt.imshow(fm)
plt.show()
#
pltmg.imsave('wheeledSim/frictionCheckerboard.png',fm)
np.save('wheeledSim/frictionCheckerboard',fm)
