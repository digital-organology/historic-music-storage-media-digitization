import numpy as np
import cv2
import os
import timeit
import math
from skimage import measure
from musicbox.helpers import gen_lut, make_color_image

## Center iterative

def center_iterative(proc):
    # Create a debug image
    if "debug_dir" in proc.parameters:
        debug_image = proc.labels.copy()
        debug_image = make_color_image(debug_image)
        debug_image = cv2.circle(debug_image, (proc.center_x, proc.center_y), 2, (255, 255, 0), 2)

    scores = []
    coords = []
    candidates = _get_candidate_points(proc)
    for candidate in candidates:
        candidate_x, candidate_y = candidate
        if "debug_dir" in proc.parameters:
            debug_image = cv2.circle(debug_image, (candidate_x, candidate_y), 2, (255, 0, 0), 2)

        score = _score_iteration(proc, candidate_x, candidate_y)
        scores.append(score)
        coords.append((candidate_x, candidate_y))

    max_idx = min(enumerate(scores), key = lambda x: x[1])[0]

    best_x, best_y = coords[max_idx]

    if proc.verbose:
        print("INFO: Best center found is (" + str(best_y) + ", " + str(best_x) + ")")

    if "debug_dir" in proc.parameters:
        debug_image = cv2.circle(debug_image, (best_x, best_y), 2, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "spiral.tiff"), debug_image)

    proc.center_x = best_x
    proc.center_y = best_y

    return True

def _get_candidate_points(proc):
    if proc.verbose:
        print("INFO: Testing " + str(proc.parameters["iterations"]) + " poins with an angle of " + str(proc.parameters["angle"] * 10) + " degrees...")

    pos = []
    base_angle = proc.parameters["angle"]
    for i in range(0, proc.parameters["iterations"]):
        angle = base_angle * i
        x_next = round(proc.center_x + (1 + 1 * angle) * math.cos(1 * angle))
        y_next = round(proc.center_y + (1 + 1 * angle) * math.sin(1 * angle))
        pos.append((x_next, y_next))
    return pos

def _score_iteration(proc, candidate_x, candidate_y):
    # We adjust the center on the processor and run the track detection
    proc.center_x = candidate_x
    proc.center_y = candidate_y
    
    # This prevents the mean_shift method to create debug information each time it is called
    # which accounts for > 50% of its runtime

    was_debug = False

    if "debug_dir" in proc.parameters:
        was_debug = True
        debug_dir = proc.parameters.pop("debug_dir")

    proc.execute_method("tracks.mean_shift")

    if was_debug:
        proc.parameters["debug_dir"] = debug_dir

    # Get the track distances

    data = np.diff(proc.track_distances, n = 1, axis = 0)

    if "debug_dir" in proc.parameters.keys():
        with open(os.path.join(proc.parameters["debug_dir"], "stat_values_iterative.csv"), "a+") as f:
            f.write(str(candidate_x) + ", " + str(candidate_y) + ", " + str(np.std(data[:,1])) + ", " + str(np.mean(data[:,1])) + "\n")

    return np.std(data[:,1])



## Extract shapes

def extract_shapes(proc):
    # This is muuuch faster than what we did previously, but there may be even faster ways
    # some more information might be found here: https://stackoverflow.com/q/30003068/3176892
    shapes = [np.argwhere(i == proc.labels) for i in np.unique(proc.labels)]
    proc.shapes = dict(zip(np.unique(proc.labels), shapes))

    # We could most likely get away with just deleting the first element (as it should always be 0)
    proc.shapes.pop(0, None)

    return True

## Center

def center_mean(proc):
    indices, counts = np.unique(proc.labels, return_counts = True)
    indices = indices[1:]
    counts = counts[1:]
    max_count = np.argmax(counts)
    outer_border_id = indices[max_count]
    
    # David, stupid and young: Transform it into a polygon
    # David, now wiser and older: This is not actually what a polygon is

    outer_border = np.argwhere(proc.labels == outer_border_id).astype(np.int32)

    # centroid = (sum(x) / len(outer_border), sum(y) / len(outer_border))
    y,x = zip(*outer_border)
    center_x, center_y = (max(x) + min(x))/2, (max(y) + min(y))/2

    if proc.verbose:
        print("Center found is: (" + str(round(center_y)) + ", " + str(round(center_x)) + ")")

    proc.center_y = round(center_y)
    proc.center_x = round(center_x)

def center_least_squares(proc):
    # First get the outer border as usual
    indices, counts = np.unique(proc.labels, return_counts = True)
    indices = indices[1:]
    counts = counts[1:]
    max_count = np.argmax(counts)
    outer_border_id = indices[max_count]

    outer_border = np.argwhere(proc.labels == outer_border_id).astype(np.int32)
    x,y = zip(*outer_border)

    x = np.array(x)
    y = np.array(y)

    coeffs = fit_ellipse(x, y)

    x0, y0 = center_from_cart(coeffs)

    if proc.verbose:
        print("Center found is: (" + str(round(x0)) + ", " + str(round(y0)) + ")")

    proc.center_x = round(x0)
    proc.center_y = round(y0)


## Labeling method

def labeling(proc):
    if proc.parameters["label_distance"] != 1:
        _legacy_label(proc)
        return True

    labels = measure.label(proc.current_image, background = 0, connectivity = 2)

    proc.labels = labels

    if "debug_dir" in proc.parameters.keys():
        start_time = timeit.default_timer()

        # Get color table
        lut = gen_lut()

        # Make sure there are at max 256 labels
        color_image = labels.copy().astype(np.uint8)
        color_image = cv2.LUT(cv2.merge((color_image, color_image, color_image)), lut)

        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "labels.png"), color_image)

        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")


    return True

def _legacy_label(proc):

    distance = proc.parameters["label_distance"]

    # Make sure every pixel either is marked as background or foreground
    img = (proc.current_image > 0).astype(int)

    # Multiply everything by -1 so that unprocessed pixels will be -1 while background is still 0
    img = img * -1

    y_limit = img.shape[0]
    x_limit = img.shape[1]

    # Iterate over every pixel once (does this make us yolo?)

    current_shape = 1

    for x_next in range(0, x_limit - 1):
        # print("Processing x:", x_next)
        for y_next in range(0, y_limit - 1):
            # print("Processing y:", y_next)
            current_shape = _process_pixel(img, y_next, x_next, distance, y_limit, x_limit, current_shape)


    if "debug_dir" in proc.parameters.keys():
        start_time = timeit.default_timer()

        # Get color table
        lut = gen_lut()

        # Make sure there are at max 256 labels
        labels = img.copy().astype(np.uint8)
        labels = cv2.LUT(cv2.merge((labels, labels, labels)), lut)

        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "labels.png"), labels)
        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")


    proc.labels = img
    return True

def _find_connected_shapes(img, y, x, distance, y_limit, x_limit):
    # Define search area

    y_from = max(0, y - distance)
    y_to = min(y_limit - 1, y + distance) + 1
    
    # We do not need to look to the right (meaning anything with higher x than what we're looking at)
    x_from = max(0, x - distance)
    x_to = min(x_limit - 1, x) + 1

    # Select subarray that is our search radius
    search_space = img[y_from:y_to, x_from:x_to]
    
    # Get ids of surrounding shapes
    ids = np.unique(search_space)
    ids = ids[ids != -1]
    ids = ids[ids != 0]

    return ids

def _process_pixel(img, y, x, distance, y_limit, x_limit, current_shape):
    pixel_content = img[y,x]

    if pixel_content == 0:
        # Background pixel, no need to do anything
        return current_shape

    # Unprocessed pixel, will need to process

    # Get all surrounding shapes
    shape_ids = _find_connected_shapes(img, y, x, distance, y_limit, x_limit)

    if shape_ids.size == 0:
        # No close shape, start a new one
        current_shape = current_shape + 1
        img[y,x] = current_shape
        return current_shape

    if shape_ids.size == 1:
        # Only one shape is close, assign this pixel to that shape
        img[y,x] = shape_ids[0]
        return current_shape

    # Multiple shapes are close. We assign this pixel to the oldest
    # meaning the one with the smallest id, and also assign all other close shapes
    # to that shape

    oldest_shape = shape_ids.min()
    img[y,x] = oldest_shape

    for shape in shape_ids[shape_ids != oldest_shape]:
        img[img == shape] = oldest_shape

    return current_shape



def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def center_from_cart(coeffs):
    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    return (x0, y0)