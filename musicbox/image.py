import numpy as np
import cv2
import os
import timeit
import random
import math
from skimage import measure
from musicbox.helpers import get_lut, make_color_image
from scipy import optimize
import circle_fit as cf

## Extract shapes

def extract_shapes(proc):
    # Most fast ways to do this only work on 1d arrays, so we flatten out the array first and get the indices for each unique element then
    # after which we stich everything back together to get 2d indices

    labels_flattend = proc.labels.ravel()
    labels_flattened_sorted = np.argsort(labels_flattend)
    keys, indices_flattend = np.unique(labels_flattend[labels_flattened_sorted], return_index=True)
    labels_ndims = np.unravel_index(labels_flattened_sorted, proc.labels.shape)
    labels_ndims = np.c_[labels_ndims] if proc.labels.ndim > 1 else labels_flattened_sorted
    indices = np.split(labels_ndims, indices_flattend[1:])
    proc.shapes = dict(zip(keys, indices))

    # We can most likely get away with just deleting the first element (as it should always be 0)
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

    # Use a least squares estimator to fit an ellipse to the coordinates and get it's coefficients

    coeffs = fit_ellipse(x, y)

    # Transform from cartesian form to (partial) general form and get the center coordinates

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
        lut = get_lut()

        # Make sure there are at max 256 labels
        color_image = labels.copy().astype(np.uint8)
        color_image = cv2.LUT(cv2.merge((color_image, color_image, color_image)), lut)

        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "labels.png"), color_image)

        np.savetxt(os.path.join(proc.parameters["debug_dir"], "label_array.txt"), labels, fmt = "%i", delimiter = "\t")

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
        lut = get_lut()

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


# The circle fitting method is adapted from here: https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

def fit_ellipse(x, y):
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


def _process_chunk(proc, start, end):
    chunk = proc.labels[start:end,:]

    shape_ids = np.unique(chunk)

    shape_ids = np.setdiff1d(shape_ids, np.array([0, proc.roll_edges[0], proc.roll_edges[1]]))

    shapes_in_chunk = {key: proc.shapes[key] for key in shape_ids if key in proc.shapes}

    left_border = np.argwhere(proc.roll_edges[0] == chunk)
    right_border = np.argwhere(proc.roll_edges[1] == chunk)

    if np.ptp(left_border, axis = 0)[1] > 5 or np.ptp(right_border, axis = 0)[1] > 5:
        print(f"WARNING: Roll edge deviates more than 5 Pixels in this chunk, notes may not be extracted correctly")

    # We prepare the configuration data we need

    # Scaling factor from our images to mm
    # We can multiply pixel distances by this factor to get to mm

    left_border_pos = left_border[:,1].min()
    right_border_pos = right_border[:,1].max()
    scaling_factor =  proc.parameters["width"] / (right_border_pos - left_border_pos)

    measurements = np.array([list(v.values()) for v in proc.parameters["track_measurements"]])
    centers = (measurements[:,1] - measurements[:,0]) / 2 + measurements[:,0]
    notes = list()

    for id, shape in shapes_in_chunk.items():
        # Will use a dedicated filtering method in the future
        if id == proc.roll_edges[0] or id == proc.roll_edges[1]:
            continue

        # We calculate the vertical height

        height = shape[:,0].max() - shape[:,0].min()

        # We first check if the shape completely fits into a track on the roll
        # Gentle reminder to myself that the coordinates are y, x

        exact_fit = measurements[(measurements[:,0] <= round((shape[:,1].min() * scaling_factor) * 2) / 2) & (measurements[:,1] >= round((shape[:,1].max() * scaling_factor) * 2) / 2)]

        if exact_fit.shape[0] == 1:
            note = np.array([id, shape[:,0].min(), height, exact_fit[0,2]])
            notes.append(note)

        # If we couldnt fit the note exactly we choose track with the closest center

        shape_center = ((shape[:,1].max() - shape[:,1].min()) / 2 + shape[:,1].min()) * scaling_factor

        idx = (np.abs(centers - shape_center)).argmin()

        if abs(shape_center - centers[idx]) > 2:
            # This went wrong
            continue

        note = np.array([id, shape[:,0].min(), height, measurements[idx,2]])
        notes.append(note)

    return notes


def extract_roll_notes(proc):
    # Notes might warp over their length so we need to calculate the relative positions of each track relative to the position of each note on the roll
    # As using too small steps might cause problems when the rolls are damaged we do this chunkwise
    # We process chunks of about 1000 pixels, looking for good points to break chunks by finding rows in out data without any hole

    chunk_beginning = 0
    left_border_id = proc.roll_edges[0]
    right_border_id = proc.roll_edges[1]
    notes = []


    while chunk_beginning < proc.labels.shape[0] - 1:
        chunk_end = chunk_beginning + 1000

        if chunk_end > proc.labels.shape[0] - 1:
            chunk_end = proc.labels.shape[0] - 1
        else:
            while not np.array_equal(np.sort(np.unique(proc.labels[chunk_end,:]), axis = 0), np.sort(np.array([0, left_border_id, right_border_id]), axis = 0)):
                chunk_end += 5
                if chunk_end > proc.labels.shape[0] - 1:
                    chunk_end = proc.labels.shape[0] - 1
                    break

        print(f"INFO: Processing Chunk from {chunk_beginning} to {chunk_end}")

        notes.extend(_process_chunk(proc, chunk_beginning, chunk_end))

        chunk_beginning = chunk_end


    proc.note_data = np.array(notes)
    return True

def filter_roll_shapes(proc):

    shapes_to_keep = []

    pixel_count = []

    # Calculate the mean pixel number over all shapes
    for key, item in proc.shapes.items():
        if item.shape[0] > proc.current_image.shape[0]:
            # this is most likely an edge of the roll
            continue

        pixel_count.append(len(item))
        
    mean_pixels = np.mean(pixel_count)

    # Actually filtering the shapes
    for key, item in proc.shapes.items():
        if item.shape[0] > proc.current_image.shape[0]:
            # this is most likely an edge of the roll
            shapes_to_keep.append(key)
            continue

        # All shapes with less pixels than the mean are most likely either artifacts or control holes which we do not use currently
        if len(item) < mean_pixels:
            continue

        # Do a very rough check if this is actually a hole by checking if it is somewhat as wide as it is heigh
        ranges = np.ptp(item, axis = 0)

        if ranges[0] == 0 or ranges[1] == 0 or round(ranges[0] / ranges[1]) < 1 or round(ranges[0] / ranges[1]) > 2:
            continue

        shapes_to_keep.append(key)

    filtered_shapes = { key_to_keep: proc.shapes[key_to_keep] for key_to_keep in shapes_to_keep}

    proc.shapes = filtered_shapes

    return True

def find_roll_edges(proc):
    # To find the edges we test a number of points and find the labels that are on that line
    # The ones most often present should be our edges
    # We then only need to check which is which

    labels = []

    lines = random.sample(range(0, proc.labels.shape[0]), 100)
    for line in lines:
        labels_on_line = np.unique(proc.labels[line,:])
        labels.extend(labels_on_line)

    idx, count = np.unique(labels, return_counts = True)

    edge_ids = []

    while len(edge_ids) < 2:
        edge_id = idx[np.argmax(count)]

        if edge_id != 0:
            edge_ids.append(edge_id)
        
        idx = np.delete(idx, np.argmax(count), 0)
        count = np.delete(count, np.argmax(count), 0)

    # Check if we got them the right way
    if proc.shapes[edge_ids[0]].mean(axis = 0)[1] > proc.shapes[edge_ids[1]].mean(axis = 0)[1]:
        edge_ids[0], edge_ids[1] = edge_ids[1], edge_ids[0]

    proc.roll_edges = edge_ids

    return True

def center_manual(proc):
    proc.center_x = proc.parameters["center_manual_x"]
    proc.center_x = proc.parameters["center_manual_y"]
    return True