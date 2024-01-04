import os

import cv2 as cv
import numpy as np

import scipy.misc

"""
https://github.com/epoch-rand/temporalProfile/blob/main/temporalProfile.py
"""

#----------------------------------------------------------------------------

# img_root = '/datasets/vimeo_90k/123'
img_root = '/data/lzk/workspace/LbasicSR/results/BasicVSR_x2_Vid4_asBI/visualization/Vid4_x3.5/crop_patch'

#----------------------------------------------------------------------------

w_point = 25                # Choose weight-coordinate to be transferred to profile
img_height = 50

h_point = 10
img_width = 100

sequence = sorted(os.listdir(img_root))

p_length = len(sequence)

profile_w = np.zeros(shape = (p_length, img_height, 3))
profile_h = np.zeros(shape = (p_length, img_width, 3))
print("Temporal Profile_w Shape:", profile_w.shape)
print("Temporal Profile_h Shape:", profile_h.shape)
print("No. of frames:", p_length)

#----------------------------------------------------------------------------

i = 0
for image in sequence:

    frame = cv.imread(os.path.join(img_root, image))

    # profile_w original
    line_w = frame[:,w_point,:]
    profile_w[i,:,:] = line_w
    
    line_h = frame[h_point, :, :]
    profile_h[i,:,:] = line_h
    

    i += 1

#----------------------------------------------------------------------------

cv.imwrite('temporalProfile_w.png', profile_w)
cv.imwrite('temporalProfile_h.png', profile_h)
print("Profile saved to disk")



