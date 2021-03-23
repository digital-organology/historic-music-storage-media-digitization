from scipy.spatial import distance
import cv2
import numpy as np
import math

def _process_shape(shape, center_x, center_y):
    # shape[:,[0, 1]] = shape[:, [1, 0]]
    rectangle = cv2.minAreaRect(shape.astype(np.float32))
    box = cv2.boxPoints(rectangle)
    box = box.astype(np.uint32)

    if len(box) != 4:
        print(len(box))
        return (0, 0)

    dists = distance.cdist(box, [[center_x, center_y]])

    dists = dists.reshape(-1)
    idx = np.argpartition(dists, 2)

    closer_points = box[idx[:2]]
    other_points = box[idx[2:]]

    # print(closer_points)
    # print(other_points)

    dists_interpoint = distance.cdist(closer_points, other_points)

    # print(dists_interpoint)

    first_line = np.row_stack((closer_points[0], other_points[dists_interpoint[0].argmin()]))
    second_line = np.row_stack((closer_points[1], other_points[dists_interpoint[1].argmin()]))

    first_m = np.array([np.mean(first_line[:,0]), np.mean(first_line[:,1])])
    second_m = np.array([np.mean(second_line[:,0]), np.mean(second_line[:,1])])
        
    # Calculate angle for first point

    third_point = np.array([center_x, first_m[1]])

    gegenkathete = distance.cdist([first_m], [third_point])[0][0]
    hypothenuse = distance.cdist([first_m], [[center_x, center_y]])[0][0]

    rads = np.arcsin(gegenkathete / hypothenuse)
    degs_first = math.degrees(rads)

    # Calculate angle for second point

    third_point = np.array([center_x, second_m[1]])

    gegenkathete = distance.cdist([second_m], [third_point])[0][0]
    hypothenuse = distance.cdist([second_m], [[center_x, center_y]])[0][0]

    g_h = gegenkathete / hypothenuse

    rads = math.asin(g_h)
    degs_second = math.degrees(rads)

    # Determine quadrant we're in to add respective 90 degrees intervals
    # Also as we are actually calculating the reverse angles if we are
    # in the upper left or lower right quadrant of the image
    # we need to take the calculated angle and subtract it from the full 90 degrees
    
    # Also we sanitize for a few special cases here where points are exactly on
    # the centers x or y coordinate

    # First point

    if first_m[0] > center_x and first_m[1] > center_y:
        degs_first = (90 - degs_first) + 90
    elif first_m[0] < center_x and first_m[1] > center_y:
        degs_first += 180
    elif first_m[0] < center_x and first_m[1] < center_y:
        degs_first = (90 - degs_first) + 270
    elif first_m[0] == center_x and first_m[1] > center_y:
        degs_first = 180
    elif first_m[0] == center_x and first_m[1] < center_y:
        degs_first = 0
    elif first_m[0] < center_x and first_m[1] == center_y:
        degs_first = 270
    elif first_m[0] > center_x and first_m[1] == center_y:
        degs_first = 90

    # Sencond point

    if second_m[0] > center_x and second_m[1] > center_y:
        degs_second = (90 - degs_second) + 90
    elif second_m[0] < center_x and second_m[1] > center_y:
        degs_second += 180
    elif second_m[0] < center_x and second_m[1] < center_y:
        degs_second = (90 - degs_second) + 270
    elif second_m[0] == center_x and second_m[1] > center_y:
        degs_second = 180
    elif second_m[0] == center_x and second_m[1] < center_y:
        degs_second = 0
    elif second_m[0] < center_x and second_m[1] == center_y:
        degs_second = 270
    elif second_m[0] > center_x and second_m[1] == center_y:
        degs_second = 90
    
    # Sanitize a few special cases

    # For the first point

    if degs_first < degs_second:
        return (degs_first, degs_second)
    else:
        return (degs_second, degs_first)

def convert_notes(shapes, shape_ids, center_x, center_y):
    shape_min = []
    shape_max = []
    for shape in shapes:
        mini, maxi = _process_shape(shape, center_x, center_y)
        shape_min.append(mini)
        shape_max.append(maxi)
    
    # import pdb; pdb.set_trace()
    arr = np.column_stack((list(shape_ids), shape_min, shape_max))
    diff = arr[:,2] - arr[:,1]
    # diff[diff > 200] = 360 - diff[diff > 200]
    arr = np.c_[arr, diff]
    return arr