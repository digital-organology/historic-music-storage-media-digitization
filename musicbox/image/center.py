import math
import cv2
import numpy as np
import os
from .notes import extract_notes


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
    available_methods = ["mean", "iterative"]
    if not any(method in s for s in available_methods):
        raise ValueError("Method " + method + " is not implemented")

    # Call apropiate method
    if method == "mean":
        return center_mean(img)
    elif method == "iterative":
        return center_iterative(img, additional_parameters)

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

def center_iterative(labels, additional_arguments):
    # import pdb; pdb.set_trace()
    max_iterations = 20
    scores = []
    coords = []
    x, y = center_mean(labels)
    candidates = _get_candidate_points(x, y, max_iterations)
    for candidate in candidates:
        candidate_x, candidate_y = candidate
        score = _score_iteration(labels, candidate_x, candidate_y, additional_arguments)
        scores.append(score)
        coords.append((candidate_x, candidate_y))

    max_idx = min(enumerate(scores), key = lambda x: x[1])[0]
    return coords[max_idx]

def _get_candidate_points(x, y, max_iterations):
    pos = []
    for i in range(0, max_iterations):
        angle = 0.5 * i
        x_next = round(x + (1 + 1 * angle) * math.cos(0.1 * angle))
        y_next = round(y + (1 + 1 * angle) * math.sin(0.1 * angle))
        pos.append((x_next, y_next))
    return pos

def _score_iteration(labels, candidate_x, candidate_y, additional_arguments):
    additional_arguments["iterative_center"] = True
    data = extract_notes(labels, candidate_x, candidate_y, additional_arguments, False)
    np.savetxt(os.path.join(additional_arguments["debug_dir"], "scores_" + str(candidate_x) + "_" + str(candidate_y) + ".csv"), data, delimiter = ",")

    with open(os.path.join(additional_arguments["debug_dir"], "stat_values.csv"), "a+") as f:
        f.write(str(candidate_x) + ", " + str(candidate_y) + ", " + str(np.std(data[:,1])) + ", " + str(np.mean(data[:,1])) + "\n")

    return np.std(data[:,1])
    # import pdb; pdb.set_trace()
