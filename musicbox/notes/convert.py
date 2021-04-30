from scipy.spatial import distance
import cv2
import numpy as np
import math
from musicbox.helpers import calculate_angles

def convert_notes(shapes, shape_ids, center_x, center_y):
    shape_min = []
    shape_max = []
    for shape in shapes:
        mini, maxi = calculate_angles(shape, center_x, center_y)
        shape_min.append(mini)
        shape_max.append(maxi)
    
    # import pdb; pdb.set_trace()
    arr = np.column_stack((list(shape_ids), shape_min, shape_max))
    diff = arr[:,2] - arr[:,1]
    # diff[diff > 200] = 360 - diff[diff > 200]
    arr = np.c_[arr, diff]
    return arr