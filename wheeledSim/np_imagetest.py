from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage

def blur(im, sigma=[3.0,2.0]):
    im = sp.ndimage.filters.gaussian_filter(im, sigma, mode='constant')
    return im

depth = Image.open('wheeledSim/wm_height_out.png')
depth = np.array(depth)/255
# depth = blur(depth)
print(depth)

import cv2
image = cv2.imread('wheeledSim/wm_height_out.png', cv2.IMREAD_UNCHANGED)
print(image)

import png
pngdata = png.Reader('wheeledSim/wm_height_out.png').read_flat()
img = np.array(pngdata[2]).reshape((pngdata[1], pngdata[0], -1))
print(img)
# # Other answer method
# im1 = Image.open('wheeledSim/wm_height_out.png').convert('L')
# im1 = np.stack((im1,)*3, axis=-1)
#
# # Your method
# im2 = Image.open('wheeledSim/wm_height_out.png')
# im2 = np.array(im2)/255
#
# # Test if identical
# print(np.array_equal(im1,im2))
# print(im1.shape)
# print(im2.shape)
# print(im2)
#
# sigma_y = 3.0
# sigma_x = 2.0
#
# plt.imshow(im2)
# plt.show()
# # Apply gaussian filter
# sigma = [sigma_y, sigma_x]
# im2 = sp.ndimage.filters.gaussian_filter(im2, sigma, mode='constant')
#
# # Display filtered array
# plt.imshow(im2)
# plt.show()

import keyboard
def handleLeftKey(e):
    print("left arrow was pressed w/ key 4")
    # work your magic

def handleLeftKeyRel(e):
    print("left arrow was pressed w/ key 4")
    # work your magic

keyboard.on_press_key("left", handleLeftKey)
keyboard.on_release_key("left", handleLeftKeyRel)

for i in range(10000000000000):
    ...
