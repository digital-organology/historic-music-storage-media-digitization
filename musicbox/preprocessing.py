import cv2
import numpy as np

def binarization(proc):
    _, img_threshold = cv2.threshold(proc.current_image, 60, 255, cv2.THRESH_BINARY)
    img_out = (img_threshold > 0).astype(np.uint8)
    proc.current_image = img_out
    return True

def edge_in(proc):
    image = (proc.current_image > 0).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)

    image_eroded = cv2.erode(image, kernel)

    edges = image - image_eroded

    proc.current_image = edges

def crop_to_contents(proc):
    y_values, x_values = np.nonzero(proc.current_image)

    y_min = y_values.min() - 20 if y_values.min() - 20 >= 0 else 0
    y_max = y_values.max() + 20 if y_values.max() + 20 <= proc.current_image.shape[0] - 1 else proc.current_image.shape[0] - 1

    x_min = x_values.min() - 20 if x_values.min() - 20 >= 0 else 0
    x_max = x_values.max() + 20 if x_values.max() + 20 <= proc.current_image.shape[1] - 1 else proc.current_image.shape[1] - 1

    proc.current_image = proc.current_image[y_min:y_max, x_min:x_max]

    return True