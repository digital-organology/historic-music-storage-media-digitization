import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy.spatial import distance

def gen_lut():
        """
        Generate a label colormap compatible with opencv lookup table, based on
        Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
        appendix C2 `Pseudocolor Generation`.
        :Returns:
            color_lut : opencv compatible color lookup table
        """

        # Blatantly stolen from here: https://stackoverflow.com/a/57080906/3176892

        tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
        arr = np.arange(256)
        r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
        g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
        b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
        return np.concatenate([[[b]], [[g]], [[r]]]).T

def make_image_from_shapes(canvas, shapes):
    bg_image = np.zeros_like(canvas).astype(np.uint16)

    for id, shape in shapes.items():
        bg_image[shape[:,0], shape[:,1]] = id
    return bg_image

def make_color_image(img):
    lut = gen_lut()

    color_image = img.astype(np.uint8)
    color_image = cv2.LUT(cv2.merge((color_image, color_image, color_image)), lut)

    return color_image

def plot_polygon(pic, polygon):
    bg_image = np.zeros_like(pic).astype(np.uint8)
    bg_image[polygon[:,1], polygon[:,0]] = 1
    plt.imshow(bg_image)
    plt.show()

def calculate_angles(shape, center_x, center_y, return_points = False):
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
        if return_points:
            return (first_m, second_m)
        else:
            return (degs_first, degs_second)
    else:
        if return_points:
            return (second_m, first_m)
        else:
            return (degs_second, degs_first)