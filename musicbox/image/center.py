import math
import cv2
import numpy as np


def detect_center(method: str, img: np.ndarray, additional_parameters = dict()):
    """Main method for the center detection methods

    Args:
        method (str): Method to use for labeling, currently only mean (a simple base method) is available
        img (numpy.ndarray): Image to detect the center on, needs to be labeled
        additional_parameters (dict, optional): Dictionary of optional arguments. Defaults to dict().

    Raises:
        TypeError: Raised if invalid input data is provided
        ValueError: Raised if an invalid method for labeling is given

    Returns:
        Tuple[int, int]: x and y coordinates of the calculcated center point 
    """

    # Basic parameter checking
    if not isinstance(method, str):
        raise TypeError("method is not a string, aborting")
    
    # See if we got a valid parameter
    available_methods = ["mean"]
    if not any(method in s for s in available_methods):
        raise ValueError("Method " + method + " is not implemented")

    # Call apropiate method
    if method == "mean":
        return center_mean(img)

    return None

def center_mean(labels):
    indices, counts = np.unique(labels, return_counts = True)
    indices = indices[1:]
    counts = counts[1:]
    max_count = np.argmax(counts)
    outer_border_id = indices[max_count]
    
    # Transform it into a polygon

    outer_border = np.argwhere(labels == outer_border_id).astype(np.int32)

    # Switch x and y around
    outer_border[:,[0, 1]] = outer_border[:, [1, 0]]

    # centroid = (sum(x) / len(outer_border), sum(y) / len(outer_border))
    y,x = zip(*outer_border)
    center_y, center_x = (max(x) + min(x))/2, (max(y) + min(y))/2

    return center_y, center_x