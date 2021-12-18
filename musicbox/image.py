import numpy as np
import cv2
import os
import timeit
from skimage import measure
from musicbox.helpers import gen_lut

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
    
    # Transform it into a polygon

    outer_border = np.argwhere(proc.labels == outer_border_id).astype(np.int32)

    # centroid = (sum(x) / len(outer_border), sum(y) / len(outer_border))
    y,x = zip(*outer_border)
    center_x, center_y = (max(x) + min(x))/2, (max(y) + min(y))/2

    proc.center_y = round(center_y)
    proc.center_x = round(center_x)

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