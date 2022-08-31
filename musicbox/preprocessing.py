import cv2
import numpy as np
import os
from itertools import groupby
from musicbox.helpers import make_color_image

def binarization(proc):
    """Binarizes the input image using threshold binarization

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method

    Returns:
        bool: True if successful false otherwise
    """

    _, img_threshold = cv2.threshold(proc.current_image, proc.parameters["bin_threshold"], 255, cv2.THRESH_BINARY)
    img_out = (img_threshold > 0).astype(np.uint8)
    proc.current_image = img_out
    return True

def edge_in(proc):
    """Applies morphological edge detection to the input image

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method

    Returns:
        bool: True if successful false otherwise
    """

    image = (proc.current_image > 0).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)

    image_eroded = cv2.erode(image, kernel)

    edges = image - image_eroded

    proc.current_image = edges

    if "debug_dir" in proc.parameters:
        edge_color = edges.copy()
        edge_color[edge_color == 1] = 3
        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "edges.tiff"), make_color_image(edge_color))
    return True

def crop_to_contents(proc):
    """Crops the input image to the content present

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method

    Returns:
        bool: True if successful false otherwise
    """

    y_values, x_values = np.nonzero(proc.current_image)

    y_min = y_values.min() - 20 if y_values.min() - 20 >= 0 else 0
    y_max = y_values.max() + 20 if y_values.max() + 20 <= proc.current_image.shape[0] - 1 else proc.current_image.shape[0] - 1

    x_min = x_values.min() - 20 if x_values.min() - 20 >= 0 else 0
    x_max = x_values.max() + 20 if x_values.max() + 20 <= proc.current_image.shape[1] - 1 else proc.current_image.shape[1] - 1

    proc.current_image = proc.current_image[y_min:y_max, x_min:x_max]

    return True

def trim_roll_ends(proc):
    """Trims an image of a piano roll to remove everything except the area actually containing holes

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method

    Returns:
        bool: True if successful false otherwise
    """

    # Count number of filled pixels per line
    counts = (proc.current_image == 1).sum(axis = 1)

    # Find all lines where more than 80% of the pixels are 1
    threshold = proc.current_image.shape[1] * 0.8
    bin_count = (counts > threshold).astype(np.uint8)

    # Find the longest sequence of these lines
    idx_pairs = np.where(np.diff(np.hstack(([False],bin_count==1,[False]))))[0].reshape(-1,2)
    bounds = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),:]

    # Trim the image
    proc.current_image = proc.current_image[bounds[0]:bounds[1], :]

    return True