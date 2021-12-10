import cv2
import numpy as np
from skimage import measure
from musicbox.helpers import gen_lut
import os


def label(method: str, img: np.ndarray, additional_parameters = dict()):
    """Main method for image labeling will dispatch the desired method

    Args:
        method (str): Method to use for labeling, currently 2-connected (via scikit-image) and n_distance (custom implementation) are available
        img (numpy.ndarray): Image to label, needs to be binarized or at least have its background be 0
        additional_parameters (dict, optional): Dictionary of optional arguments. Defaults to dict().

    Raises:
        TypeError: Raised if invalid input data is provided
        ValueError: Raised if an invalid method for labeling is given

    Returns:
        numpy.ndarray: Image with labeled components
    """

    # Basic parameter checking
    if not isinstance(method, str):
        raise TypeError("method is not a string, aborting")
    
    # See if we got a valid parameter
    available_methods = ["2-connected", "n-distance"]
    if not any(method in s for s in available_methods):
        raise ValueError("Method " + method + " is not implemented")

    # Call apropiate method
    if method == "2-connected":
        return label_2_connected(img, additional_parameters)
    if method == "n-distance":
        return label_n_distance(img, additional_parameters)

    return None

def label_2_connected(img, additional_parameters):
    img = measure.label(img, background = 0, connectivity = 2)


    if "debug_dir" in additional_parameters:
        # Get color table
        lut = gen_lut()

        # Make sure there are at max 256 labels
        labels = img.copy().astype(np.uint8)
        labels = cv2.LUT(cv2.merge((labels, labels, labels)), lut)

        cv2.imwrite(os.path.join(additional_parameters["debug_dir"], "labels.png"), labels)

    return img
        

def label_n_distance(img, additional_parameters):

    if not "distance" in additional_parameters:
        distance = 5
    else:
        distance = additional_parameters.distance

    # Make sure every pixel either is marked as background or foreground
    img = img > 0

    img = img.astype(int)


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


    if "debug_dir" in additional_parameters:
        # Get color table
        lut = gen_lut()

        # Make sure there are at max 256 labels
        labels = img.copy().astype(np.uint8)
        labels = cv2.LUT(cv2.merge((labels, labels, labels)), lut)

        cv2.imwrite(os.path.join(additional_parameters["debug_dir"], "labels.png"), labels)

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